# app.py (Added Volume Gestures - April 24, 2025)

import cv2
import logging
import configparser
from collections import deque, Counter
import pyautogui
import time
import os
import threading
import base64
import io
from PIL import Image

from flask import Flask, render_template, Response, request # Added request for sid
from flask_socketio import SocketIO, emit

try:
    from playsound import playsound
    SOUND_ENABLED = True
except ImportError: SOUND_ENABLED = False; logging.warning("playsound not found.")
except Exception as e: SOUND_ENABLED = False; logging.warning(f"playsound import error: {e}")

try:
    from camera import Camera
    from gesture import GestureRecognizer # Using updated gesture.py
    import mediapipe as mp
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
except ImportError as e: print(f"Error importing local modules: {e}"); exit()

# --- Logging, Config Loading (Same as before) ---
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s [%(threadName)s] - [%(module)s:%(funcName)s] - %(message)s')
config = configparser.ConfigParser(); config_file = 'config.ini'
if not os.path.exists(config_file): logging.error(f"Config file '{config_file}' not found!"); exit()
try:
    config.read(config_file)
    logging.info(f"Loaded config from {config_file}")
    # Load all config values...
    CAMERA_INDEX = config.getint('General', 'CameraIndex', fallback=0)
    WINDOW_NAME = config.get('General', 'WindowName', fallback='Gesture Control Web')
    DETECTION_CONFIDENCE = config.getfloat('MediaPipe', 'DetectionConfidence', fallback=0.7)
    TRACKING_CONFIDENCE = config.getfloat('MediaPipe', 'TrackingConfidence', fallback=0.7)
    GESTURE_HISTORY_MAX_LEN = config.getint('Control', 'GestureHistoryMaxLen', fallback=15)
    GESTURE_CONFIRM_THRESHOLD = config.getint('Control', 'GestureConfirmThreshold', fallback=10)
    ACTION_COOLDOWN = config.getfloat('Control', 'ActionCooldown', fallback=1.5)
    VISUAL_FEEDBACK_DURATION = config.getfloat('Control', 'VisualFeedbackDuration', fallback=1.5)
    ACTIVATION_GESTURE = config.get('Control', 'ActivationGesture', fallback='OK Sign')
    ACTIVATION_TIMEOUT = config.getfloat('Control', 'ActivationTimeout', fallback=10.0)
    if GESTURE_CONFIRM_THRESHOLD > GESTURE_HISTORY_MAX_LEN: GESTURE_CONFIRM_THRESHOLD = GESTURE_HISTORY_MAX_LEN; logging.warning("Confirm threshold adjusted.")
except Exception as e: logging.error(f"Error reading config: {e}. Exiting."); exit()

# --- Sound File Paths & Helper (Same as before) ---
SOUNDS_FOLDER = 'sounds'; SOUND_ACTION_PATH = os.path.join(SOUNDS_FOLDER, 'action.wav'); SOUND_STATUS_PATH = os.path.join(SOUNDS_FOLDER, 'status.wav')
# (Add file existence checks as before)
def play_sound_async(sound_path):
    if SOUND_ENABLED and os.path.exists(sound_path):
        try: threading.Thread(target=playsound, args=(sound_path,), daemon=True).start()
        except Exception as e: logging.error(f"Error playing sound '{sound_path}': {e}", exc_info=False)
    elif SOUND_ENABLED: logging.debug(f"Sound file not found: {sound_path}")

# --- Drawing Specs (Same as before) ---
DEFAULT_LANDMARK_SPEC=mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2)
DEFAULT_CONNECTION_SPEC=mp_drawing.DrawingSpec(color=(0,0,255), thickness=2)
HIGHLIGHT_LANDMARK_SPEC=mp_drawing.DrawingSpec(color=(0,255,255), thickness=3, circle_radius=3)
HIGHLIGHT_CONNECTION_SPEC=mp_drawing.DrawingSpec(color=(255,255,255), thickness=3)

# --- Flask & SocketIO Setup (Same as before) ---
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key_here!'
socketio = SocketIO(app, async_mode='eventlet')

# --- Global State Variables (Same as before) ---
app_state = { "gesture_histories": {}, "current_stable_gestures": {}, "last_action_time": 0, "is_active": False, "last_hand_detected_time": time.time(), "current_feedback_info": {'message': "", 'time': 0, 'hand': None}, "running": True, "last_emitted_status": None }
state_lock = threading.Lock()
video_thread = None


# ***** UPDATE GESTURE_ACTION_MAP *****
GESTURE_ACTION_MAP = {
    "Open Palm":   {'func': pyautogui.press, 'args': ['playpause'],  'feedback': 'Play/Pause Media'},
    "Fist":        {'func': pyautogui.press, 'args': ['playpause'],  'feedback': 'Play/Pause Media'},
    "Thumbs Up":   {'func': pyautogui.press, 'args': ['volumeup'],   'feedback': 'Volume Up'},
    "Thumbs Down": {'func': pyautogui.press, 'args': ['volumedown'], 'feedback': 'Volume Down'},
    # Add more gestures here later if needed
}
# *************************************


# --- Background Thread (video_processing_thread) ---
# (Keep the existing video_processing_thread function structure.
# No changes are needed inside it, as it already uses GESTURE_ACTION_MAP
# dynamically. Just make sure it's using the updated map defined above.)
def video_processing_thread():
    logging.info("Video processing thread started.")
    cam = Camera(camera_index=CAMERA_INDEX)
    try: cam.start_camera()
    except IOError as e:
        logging.error(f"Cam Error: {e}")
        with state_lock:
            app_state["running"] = False
        socketio.emit('server_error', {'message': f'Cam Error {CAMERA_INDEX}'})
        return

    recognizer = GestureRecognizer( min_detection_confidence=DETECTION_CONFIDENCE, min_tracking_confidence=TRACKING_CONFIDENCE)
    logging.info(f"Recognizer initialized. Confidence Detect={DETECTION_CONFIDENCE}, Track={TRACKING_CONFIDENCE}")

    while app_state["running"]:
        with state_lock: # Lock for accessing/modifying shared state
            current_time = time.time()
            is_active = app_state["is_active"]
            last_action_time = app_state["last_action_time"]
            feedback_info = app_state["current_feedback_info"].copy()

            # --- Capture & Process ---
            frame = cam.capture_frame()
            if frame is None: socketio.sleep(0.1); continue # Use socketio sleep
            processed_hands_data = recognizer.recognize_gestures(frame)
            hand_detected_this_frame = bool(processed_hands_data)
            if hand_detected_this_frame: app_state["last_hand_detected_time"] = current_time

            # --- Smoothing, Activation/Deactivation, Action Triggering ---
            # (This whole section remains logically the same as before,
            # it will now correctly identify Thumbs Up/Down if gesture.py returns them,
            # check if they are in the updated GESTURE_ACTION_MAP, and trigger
            # the corresponding action ('volumeup'/'volumedown') if active
            # and cooldown allows)

            detected_hands_this_frame_keys = set()
            current_gestures_display = {}
            newly_stable_actionable_gestures = {}

            for hand_data in processed_hands_data:
                hand_key = hand_data.get('handedness', 'Unknown')
                current_raw_gesture = hand_data.get('gesture', 'Undefined')
                detected_hands_this_frame_keys.add(hand_key)
                if hand_key not in app_state["gesture_histories"]: app_state["gesture_histories"][hand_key] = deque(maxlen=GESTURE_HISTORY_MAX_LEN); app_state["current_stable_gestures"][hand_key] = "Detecting..."
                app_state["gesture_histories"][hand_key].append(current_raw_gesture)
                last_stable_state = app_state["current_stable_gestures"].get(hand_key, "Detecting...")

                current_confirmed_stable = None
                if len(app_state["gesture_histories"][hand_key]) >= GESTURE_CONFIRM_THRESHOLD:
                    try:
                        counts = Counter(app_state["gesture_histories"][hand_key])
                        most_common, count = counts.most_common(1)[0]
                        if count >= GESTURE_CONFIRM_THRESHOLD and isinstance(most_common, str) and most_common not in ["Detecting...", "Gesture Undefined", "Analysis Error", "Undefined"]:
                            if app_state["current_stable_gestures"].get(hand_key) != most_common: app_state["current_stable_gestures"][hand_key] = most_common; current_confirmed_stable = most_common; logging.debug(f"Stable confirmed/changed: {hand_key}, {most_common}")
                            else: current_confirmed_stable = most_common
                    except IndexError: pass

                # --- Activation ---
                if current_confirmed_stable:
                    if current_confirmed_stable == ACTIVATION_GESTURE and not is_active:
                        app_state["is_active"] = True; app_state["current_feedback_info"] = {'message': "System Activated", 'time': current_time, 'hand': hand_key}; play_sound_async(SOUND_STATUS_PATH); app_state["last_action_time"] = current_time; logging.info(f"System ACTIVATED by {ACTIVATION_GESTURE} ({hand_key})."); is_active = True

                # --- Store Actionable ---
                current_stable_state_for_action = app_state["current_stable_gestures"].get(hand_key)
                if current_stable_state_for_action in GESTURE_ACTION_MAP: # Check against updated map
                    if current_stable_state_for_action != last_stable_state: newly_stable_actionable_gestures[hand_key] = current_stable_state_for_action; logging.info(f"Actionable stable: {hand_key}, {current_stable_state_for_action}")
                current_gestures_display[hand_key] = f"{hand_key}: {app_state['current_stable_gestures'].get(hand_key, 'Detecting...')}"

            # --- Cleanup ---
            hands_to_remove = set(app_state["gesture_histories"].keys()) - detected_hands_this_frame_keys
            for hand_key in hands_to_remove:
                if hand_key in app_state["current_stable_gestures"]: del app_state["current_stable_gestures"][hand_key]
                if hand_key in app_state["gesture_histories"]: del app_state["gesture_histories"][hand_key]

            # --- Deactivation ---
            time_since_last_hand = current_time - app_state["last_hand_detected_time"]
            if is_active and time_since_last_hand > ACTIVATION_TIMEOUT:
                app_state["is_active"] = False; app_state["current_feedback_info"] = {'message': "System Deactivated (Timeout)", 'time': current_time, 'hand': None}; play_sound_async(SOUND_STATUS_PATH); logging.info(f"System DEACTIVATED (Timeout > {ACTIVATION_TIMEOUT}s)."); is_active = False

            # --- Action Triggering ---
            if is_active and newly_stable_actionable_gestures and (current_time - last_action_time > ACTION_COOLDOWN):
                triggering_hand_key = list(newly_stable_actionable_gestures.keys())[0]
                triggering_gesture = newly_stable_actionable_gestures[triggering_hand_key]
                if triggering_gesture in GESTURE_ACTION_MAP: # Now includes Thumbs Up/Down
                    action_details = GESTURE_ACTION_MAP[triggering_gesture]
                    try:
                        logging.info(f"ACTION TRIGGERED: {triggering_hand_key}, {triggering_gesture} => {action_details['feedback']}")
                        action_details['func'](*action_details['args']) # Execute action
                        app_state["current_feedback_info"] = {'message': f"Action: {action_details['feedback']}", 'time': current_time, 'hand': triggering_hand_key}
                        play_sound_async(SOUND_ACTION_PATH) # Play action sound
                        app_state["last_action_time"] = current_time # Reset cooldown
                    except Exception as e: logging.error(f"Error executing action for {triggering_gesture}: {e}")

            # --- Drawing & Encoding Frame ---
            if current_time - feedback_info['time'] > VISUAL_FEEDBACK_DURATION: feedback_info = {'message': "", 'time': 0, 'hand': None}; app_state["current_feedback_info"] = feedback_info
            hand_to_highlight = feedback_info.get('hand')
            final_frame = frame.copy()
            if processed_hands_data: # Draw landmarks
                for hand_data in processed_hands_data:
                    landmarks = hand_data.get('landmarks'); handedness = hand_data.get('handedness')
                    if not landmarks: continue
                    spec = (HIGHLIGHT_LANDMARK_SPEC, HIGHLIGHT_CONNECTION_SPEC) if handedness == hand_to_highlight else (DEFAULT_LANDMARK_SPEC, DEFAULT_CONNECTION_SPEC)
                    mp_drawing.draw_landmarks(image=final_frame, landmark_list=landmarks, connections=mp_hands.HAND_CONNECTIONS, landmark_drawing_spec=spec[0], connection_drawing_spec=spec[1])
            # Draw text overlays
            status_text = "Status: ACTIVE" if is_active else f"Status: INACTIVE (Show {ACTIVATION_GESTURE})"
            status_color = (0, 255, 0) if is_active else (0, 0, 255)
            cv2.putText(final_frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2, cv2.LINE_AA)
            feedback_message_to_display = feedback_info.get('message', "")
            if feedback_message_to_display:
                feedback_color = (0, 255, 255)
                if "Activated" in feedback_message_to_display or "Deactivated" in feedback_message_to_display: feedback_color = status_color
                try: cv2.putText(final_frame, feedback_message_to_display, (10, final_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, feedback_color, 2, cv2.LINE_AA)
                except Exception as e: logging.error(f"Error drawing feedback text: {e}")

            # Encode frame
            try:
                img_rgb = cv2.cvtColor(final_frame, cv2.COLOR_BGR2RGB); pil_img = Image.fromarray(img_rgb)
                buffer = io.BytesIO(); pil_img.save(buffer, format="JPEG")
                base64_frame = base64.b64encode(buffer.getvalue()).decode('utf-8')
                # --- Emit Data via SocketIO ---
                socketio.emit('video_frame', {'frame': base64_frame})
                current_status_data = { 'is_active': is_active, 'activation_gesture_name': ACTIVATION_GESTURE, 'detected_gestures': current_gestures_display, 'feedback': feedback_info }
                if current_status_data != app_state["last_emitted_status"]: socketio.emit('update_status', current_status_data); app_state["last_emitted_status"] = current_status_data
            except Exception as e: logging.error(f"Error encoding/emitting frame: {e}", exc_info=False)

        socketio.sleep(0.01) # Yield

    # Cleanup
    if cam.is_opened(): cam.release()
    logging.info("Video processing thread finished.")


# --- Flask Routes & SocketIO Events (Same as before) ---
@app.route('/')
def index():
    return render_template('index.html', window_title=WINDOW_NAME)

@socketio.on('connect')
def handle_connect():
    logging.info(f"Client connected: {request.sid}")
    global video_thread
    if video_thread is None or not video_thread.is_alive():
        logging.info("Starting background video processing thread.")
        with state_lock: app_state["running"] = True
        video_thread = socketio.start_background_task(target=video_processing_thread)

@socketio.on('disconnect')
def handle_disconnect():
    logging.info(f"Client disconnected: {request.sid}")

# (Keep handle_set_active function if you want client control)
# ...

if __name__ == '__main__':
    logging.info("Starting Flask-SocketIO server...")
    print(f"\nOpen web browser to: http://127.0.0.1:5000\nCTRL+C to stop.\n")
    socketio.run(app, host='127.0.0.1', port=5000, debug=False, use_reloader=False)
    logging.info("Server stopping...")
    with state_lock: app_state["running"] = False
    if video_thread: video_thread.join(timeout=2)
    logging.info("Server stopped.")