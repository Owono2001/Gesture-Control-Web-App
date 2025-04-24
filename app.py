# app.py (Web Application Backend)

import cv2
import logging
import configparser
from collections import deque, Counter
import pyautogui
import time
import os
import threading
import base64 # To encode frames for web display
import io # To handle image bytes
from PIL import Image # To convert numpy array to image bytes

# --- Web Framework & WebSockets ---
from flask import Flask, render_template, Response, request
from flask_socketio import SocketIO, emit

# --- Sound ---
try:
    from playsound import playsound
    SOUND_ENABLED = True
except ImportError:
    logging.warning("playsound library not found. Audio cues disabled.")
    SOUND_ENABLED = False
except Exception as e:
    logging.warning(f"Error importing playsound: {e}. Audio cues disabled.")
    SOUND_ENABLED = False

# --- Custom Modules ---
try:
    from camera import Camera # Using simplified camera.py
    from gesture import GestureRecognizer # Using your existing gesture.py
    import mediapipe as mp # Need mediapipe for drawing now
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
except ImportError as e:
    print(f"Error importing local modules: {e}")
    exit()

# --- Basic Logging Setup ---
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s [%(threadName)s] - [%(module)s:%(funcName)s] - %(message)s')

# --- Configuration Loading ---
config = configparser.ConfigParser()
config_file = 'config.ini'
# (Same config loading logic as in previous main.py, using try/except/fallback)
if not os.path.exists(config_file):
    logging.error(f"CRITICAL ERROR: Configuration file '{config_file}' not found!")
    exit()
try:
    config.read(config_file)
    logging.info(f"Loaded configuration from {config_file}")
    CAMERA_INDEX = config.getint('General', 'CameraIndex', fallback=0)
    WINDOW_NAME = config.get('General', 'WindowName', fallback='Gesture Control Web') # Default for title maybe
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

# --- Sound File Paths & Helper ---
SOUNDS_FOLDER = 'sounds'; SOUND_ACTION_PATH = os.path.join(SOUNDS_FOLDER, 'action.wav'); SOUND_STATUS_PATH = os.path.join(SOUNDS_FOLDER, 'status.wav')
if SOUND_ENABLED and not os.path.exists(SOUNDS_FOLDER): logging.warning(f"Sounds folder '{SOUNDS_FOLDER}' not found.")
if SOUND_ENABLED and not os.path.exists(SOUND_ACTION_PATH): logging.warning(f"Action sound not found: {SOUND_ACTION_PATH}")
if SOUND_ENABLED and not os.path.exists(SOUND_STATUS_PATH): logging.warning(f"Status sound not found: {SOUND_STATUS_PATH}")

def play_sound_async(sound_path):
    if SOUND_ENABLED and os.path.exists(sound_path):
        try: threading.Thread(target=playsound, args=(sound_path,), daemon=True).start()
        except Exception as e: logging.error(f"Error playing sound '{sound_path}': {e}", exc_info=False)
    elif SOUND_ENABLED: logging.debug(f"Sound file not found, skipping: {sound_path}")

# --- Gesture Action Map ---
GESTURE_ACTION_MAP = {
    "Open Palm": {'func': pyautogui.press, 'args': ['playpause'], 'feedback': 'Play Media'},
    "Fist":      {'func': pyautogui.press, 'args': ['playpause'], 'feedback': 'Pause Media'},
}

# --- Drawing Specs (Mirrors camera.py logic, but now used here) ---
DEFAULT_LANDMARK_SPEC = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
DEFAULT_CONNECTION_SPEC = mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
HIGHLIGHT_LANDMARK_SPEC = mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=3, circle_radius=3)
HIGHLIGHT_CONNECTION_SPEC = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=3)

# --- Flask & SocketIO Setup ---
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key_here!' # Change this for production
# Use eventlet for async mode, crucial for background tasks + SocketIO
socketio = SocketIO(app, async_mode='eventlet')

# --- Global State Variables (Managed by the background thread) ---
# Use a dictionary to group state for easier management potentially later
app_state = {
    "gesture_histories": {},
    "current_stable_gestures": {},
    "last_action_time": 0,
    "is_active": False,
    "last_hand_detected_time": time.time(),
    "current_feedback_info": {'message': "", 'time': 0, 'hand': None},
    "running": True, # Flag to stop the background thread
    "last_emitted_status": None # Track emitted status to avoid redundant emits
}
state_lock = threading.Lock() # To protect access to shared state if needed (good practice)

# --- Background Thread for Camera Processing ---
video_thread = None

def video_processing_thread():
    """Handles camera capture, gesture recognition, state management, and emitting data."""
    logging.info("Video processing thread started.")

    cam = Camera(camera_index=CAMERA_INDEX)
    try:
        cam.start_camera()
    except IOError as e:
        logging.error(f"Failed to start camera in background thread: {e}")
        with state_lock: app_state["running"] = False
        socketio.emit('server_error', {'message': f'Failed to start camera {CAMERA_INDEX}'})
        return # Stop thread if camera fails

    recognizer = GestureRecognizer(
        min_detection_confidence=DETECTION_CONFIDENCE,
        min_tracking_confidence=TRACKING_CONFIDENCE
    )

    frame_count = 0
    start_time = time.time()

    while app_state["running"]:
        with state_lock: # Ensure thread-safe access/modification of state
            # Make local copies of state needed for this iteration
            current_time = time.time()
            is_active = app_state["is_active"]
            last_action_time = app_state["last_action_time"]
            last_hand_detected_time = app_state["last_hand_detected_time"]
            feedback_info = app_state["current_feedback_info"].copy()

            # --- Capture and Process ---
            frame = cam.capture_frame()
            if frame is None:
                logging.warning("No frame captured, sleeping briefly.")
                socketio.sleep(0.1) # Use socketio.sleep in async_mode='eventlet'
                continue

            processed_hands_data = recognizer.recognize_gestures(frame)
            hand_detected_this_frame = bool(processed_hands_data)

            if hand_detected_this_frame:
                app_state["last_hand_detected_time"] = current_time # Update shared state

            # --- Gesture Smoothing & Activation ---
            detected_hands_this_frame_keys = set()
            current_gestures_display = {} # For status emission
            newly_stable_actionable_gestures = {} # Reset each frame

            for hand_data in processed_hands_data:
                hand_key = hand_data.get('handedness', 'Unknown')
                current_raw_gesture = hand_data.get('gesture', 'Undefined')
                detected_hands_this_frame_keys.add(hand_key)

                if hand_key not in app_state["gesture_histories"]:
                    app_state["gesture_histories"][hand_key] = deque(maxlen=GESTURE_HISTORY_MAX_LEN)
                    app_state["current_stable_gestures"][hand_key] = "Detecting..."

                app_state["gesture_histories"][hand_key].append(current_raw_gesture)
                last_stable_state = app_state["current_stable_gestures"].get(hand_key, "Detecting...")

                current_confirmed_stable = None
                if len(app_state["gesture_histories"][hand_key]) >= GESTURE_CONFIRM_THRESHOLD:
                    try:
                        counts = Counter(app_state["gesture_histories"][hand_key])
                        most_common, count = counts.most_common(1)[0]
                        if count >= GESTURE_CONFIRM_THRESHOLD and isinstance(most_common, str) and most_common not in ["Detecting...", "Gesture Undefined", "Analysis Error", "Undefined"]:
                            if app_state["current_stable_gestures"].get(hand_key) != most_common:
                                app_state["current_stable_gestures"][hand_key] = most_common
                                current_confirmed_stable = most_common
                                logging.debug(f"Stable confirmed/changed: {hand_key}, {most_common}")
                            else:
                                current_confirmed_stable = most_common
                    except IndexError: pass # Should not happen

                # --- Activation Logic ---
                if current_confirmed_stable:
                    if current_confirmed_stable == ACTIVATION_GESTURE and not is_active:
                        app_state["is_active"] = True
                        app_state["current_feedback_info"] = {'message': "System Activated", 'time': current_time, 'hand': hand_key}
                        play_sound_async(SOUND_STATUS_PATH)
                        app_state["last_action_time"] = current_time # Reset cooldown
                        logging.info(f"System ACTIVATED by {ACTIVATION_GESTURE} ({hand_key}).")
                        is_active = True # Update local copy for this iteration

                # Store Actionable Gestures that JUST became stable
                current_stable_state_for_action = app_state["current_stable_gestures"].get(hand_key)
                if current_stable_state_for_action in GESTURE_ACTION_MAP:
                    if current_stable_state_for_action != last_stable_state:
                        newly_stable_actionable_gestures[hand_key] = current_stable_state_for_action
                        logging.info(f"Actionable stable: {hand_key}, {current_stable_state_for_action}")

                current_gestures_display[hand_key] = f"{hand_key}: {app_state['current_stable_gestures'].get(hand_key, 'Detecting...')}"

            # --- Cleanup histories ---
            hands_to_remove = set(app_state["gesture_histories"].keys()) - detected_hands_this_frame_keys
            for hand_key in hands_to_remove:
                if hand_key in app_state["current_stable_gestures"]: del app_state["current_stable_gestures"][hand_key]
                if hand_key in app_state["gesture_histories"]: del app_state["gesture_histories"][hand_key]

            # --- Deactivation Logic ---
            time_since_last_hand = current_time - app_state["last_hand_detected_time"]
            if is_active and time_since_last_hand > ACTIVATION_TIMEOUT:
                app_state["is_active"] = False
                app_state["current_feedback_info"] = {'message': "System Deactivated (Timeout)", 'time': current_time, 'hand': None}
                play_sound_async(SOUND_STATUS_PATH)
                logging.info(f"System DEACTIVATED (Timeout > {ACTIVATION_TIMEOUT}s).")
                is_active = False # Update local copy

            # --- Action Triggering ---
            if is_active and newly_stable_actionable_gestures and (current_time - last_action_time > ACTION_COOLDOWN):
                triggering_hand_key = list(newly_stable_actionable_gestures.keys())[0]
                triggering_gesture = newly_stable_actionable_gestures[triggering_hand_key]
                if triggering_gesture in GESTURE_ACTION_MAP:
                    action_details = GESTURE_ACTION_MAP[triggering_gesture]
                    try:
                        logging.info(f"ACTION TRIGGERED: {triggering_hand_key}, {triggering_gesture} => {action_details['feedback']}")
                        action_details['func'](*action_details['args']) # Execute pyautogui
                        app_state["current_feedback_info"] = {'message': f"Action: {action_details['feedback']}", 'time': current_time, 'hand': triggering_hand_key}
                        play_sound_async(SOUND_ACTION_PATH)
                        app_state["last_action_time"] = current_time # Update shared state
                    except Exception as e: logging.error(f"Error executing action for {triggering_gesture}: {e}")


            # --- Drawing & Encoding Frame ---
            # Check feedback duration
            if current_time - feedback_info['time'] > VISUAL_FEEDBACK_DURATION:
                feedback_info = {'message': "", 'time': 0, 'hand': None} # Clear feedback
                app_state["current_feedback_info"] = feedback_info # Update shared state

            hand_to_highlight = feedback_info.get('hand')
            final_frame = frame.copy() # Work on a copy

            # Draw landmarks with potential highlighting
            if processed_hands_data:
                for hand_data in processed_hands_data:
                    landmarks = hand_data.get('landmarks')
                    handedness = hand_data.get('handedness')
                    if not landmarks: continue
                    spec = (HIGHLIGHT_LANDMARK_SPEC, HIGHLIGHT_CONNECTION_SPEC) if handedness == hand_to_highlight else (DEFAULT_LANDMARK_SPEC, DEFAULT_CONNECTION_SPEC)
                    mp_drawing.draw_landmarks(image=final_frame, landmark_list=landmarks, connections=mp_hands.HAND_CONNECTIONS, landmark_drawing_spec=spec[0], connection_drawing_spec=spec[1])

            # Add Status/Feedback text overlays (similar to desktop version)
            # Status Text
            status_text = "Status: ACTIVE" if is_active else f"Status: INACTIVE (Show {ACTIVATION_GESTURE})"
            status_color = (0, 255, 0) if is_active else (0, 0, 255)
            cv2.putText(final_frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2, cv2.LINE_AA)
            # Feedback Text
            feedback_message_to_display = feedback_info.get('message', "")
            if feedback_message_to_display:
                feedback_color = (0, 255, 255)
                if "Activated" in feedback_message_to_display or "Deactivated" in feedback_message_to_display: feedback_color = status_color
                try:
                    # Simplified text placement
                     cv2.putText(final_frame, feedback_message_to_display, (10, final_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, feedback_color, 2, cv2.LINE_AA)
                except Exception as e: logging.error(f"Error drawing feedback text: {e}")


            # Encode frame as JPEG base64 for web
            try:
                # Convert frame to PIL Image
                img_rgb = cv2.cvtColor(final_frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(img_rgb)
                # Save to a byte stream
                buffer = io.BytesIO()
                pil_img.save(buffer, format="JPEG")
                base64_frame = base64.b64encode(buffer.getvalue()).decode('utf-8')

                # --- Emit Data via SocketIO ---
                # Emit video frame
                socketio.emit('video_frame', {'frame': base64_frame})

                # Emit status updates (only if changed to reduce traffic?)
                current_status_data = {
                    'is_active': is_active,
                    'activation_gesture_name': ACTIVATION_GESTURE,
                    'detected_gestures': current_gestures_display, # Dict {'Left': 'Fist', 'Right': 'Open Palm'}
                    'feedback': feedback_info
                }
                # Optimization: Only emit if status data actually changed
                if current_status_data != app_state["last_emitted_status"]:
                    socketio.emit('update_status', current_status_data)
                    app_state["last_emitted_status"] = current_status_data

            except Exception as e:
                logging.error(f"Error encoding/emitting frame: {e}", exc_info=False)


        # --- Calculate and log FPS periodically ---
        frame_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time > 5: # Log FPS every 5 seconds
            fps = frame_count / elapsed_time
            logging.debug(f"Processing FPS: {fps:.2f}")
            frame_count = 0
            start_time = time.time()

        # Yield control, essential for eventlet/socketio
        socketio.sleep(0.01) # Sleep for ~10ms (adjust as needed)

    # --- Cleanup ---
    if cam.is_opened():
        cam.release()
    logging.info("Video processing thread finished.")


# --- Flask Routes ---
@app.route('/')
def index():
    """Serve the main HTML page."""
    return render_template('index.html', window_title=WINDOW_NAME) # Pass title

# --- SocketIO Events ---
@socketio.on('connect')
def handle_connect():
    """Handle new client connection."""
    logging.info(f"Client connected: {request.sid}") # Use request.sid from flask
    global video_thread
    # Start the background thread only if it doesn't exist or is dead
    if video_thread is None or not video_thread.is_alive():
        logging.info("Starting background video processing thread.")
        # Reset running flag in case it was stopped
        with state_lock: app_state["running"] = True
        # Use socketio.start_background_task for better integration
        video_thread = socketio.start_background_task(target=video_processing_thread)
        # Old way (less ideal with eventlet/gevent):
        # video_thread = threading.Thread(target=video_processing_thread, daemon=True)
        # video_thread.start()

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    logging.info(f"Client disconnected: {request.sid}")
    # Optionally stop the thread if no clients are connected?
    # Or let it run until the server stops. For simplicity, let it run.

# Example: Handle command from client (optional)
@socketio.on('set_active_state')
def handle_set_active(data):
    if isinstance(data, dict) and 'active' in data:
        new_state = bool(data['active'])
        with state_lock:
            app_state['is_active'] = new_state
        logging.info(f"Active state set to {new_state} by client {request.sid}")
        # Emit status update immediately
        socketio.emit('update_status', {
             'is_active': new_state,
             'activation_gesture_name': ACTIVATION_GESTURE,
             'detected_gestures': app_state['current_stable_gestures'],
             'feedback': {'message': f"Status set to {new_state}", 'time': time.time(), 'hand': None}
        }, broadcast=True) # Broadcast to all clients

if __name__ == '__main__':
    logging.info("Starting Flask-SocketIO server...")
    print("\nGesture Control Web App")
    print("-----------------------")
    print(f"Open your web browser and go to: http://127.0.0.1:5000")
    print("-----------------------")
    print("Press CTRL+C in this terminal to stop the server.")
    # Use socketio.run for proper integration with eventlet/gevent
    # host='0.0.0.0' makes it accessible on your network, use '127.0.0.1' for local only
    socketio.run(app, host='127.0.0.1', port=5000, debug=False, use_reloader=False)
    # When server stops (Ctrl+C):
    logging.info("Server stopping...")
    with state_lock: app_state["running"] = False # Signal thread to stop
    if video_thread: video_thread.join(timeout=2) # Wait briefly for thread cleanup
    logging.info("Server stopped.")