# gesture.py (Robust Thumbs Up/Down using Vectors - April 24, 2025)
import cv2
import mediapipe as mp
import math
import logging
import numpy as np # Import numpy for vector operations

HandLandmark = mp.solutions.hands.HandLandmark

class GestureRecognizer:
    """
    Recognizes hand gestures using vector math for more robust Thumbs Up/Down.
    Includes OK Sign (Activation), Palm/Fist (Play/Pause).
    """
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=2):
        self.mp_hands = mp.solutions.hands
        try:
            self.hands = self.mp_hands.Hands(
                static_image_mode=False, max_num_hands=max_num_hands,
                min_detection_confidence=min_detection_confidence, min_tracking_confidence=min_tracking_confidence
            )
            logging.info("MediaPipe Hands initialized successfully.")
        except Exception as e: logging.error(f"Failed to initialize MediaPipe Hands: {e}", exc_info=True); raise

    # --- Vector Helper Functions ---
    def _get_vector(self, p1, p2):
        """Calculates the vector from p1 to p2."""
        if p1 is None or p2 is None: return None
        return np.array([p2.x - p1.x, p2.y - p1.y, p2.z - p1.z])

    def _normalize_vector(self, v):
        """Normalizes a vector to unit length."""
        if v is None: return None
        norm = np.linalg.norm(v)
        if norm == 0: return v # Avoid division by zero
        return v / norm

    def _dot_product(self, v1, v2):
        """Calculates the dot product of two vectors."""
        if v1 is None or v2 is None: return 0 # Return neutral value if vector is missing
        return np.dot(v1, v2)

    def _calculate_distance(self, p1, p2):
        # (Same as before)
        if p1 is None or p2 is None: return float('inf')
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

    # --- Basic Finger State Helpers ---
    def _is_finger_extended(self, finger_name: str, tip_landmark, pip_landmark, mcp_landmark):
        # (Same basic logic as before, logging shortened)
        if tip_landmark is None or pip_landmark is None or mcp_landmark is None: return False
        is_basic_extended = (tip_landmark.y < pip_landmark.y < mcp_landmark.y)
        logging.debug(f"[{finger_name}_EXT_CHECK] -> BasicExt={is_basic_extended}")
        return is_basic_extended

    def _is_finger_curled(self, finger_name: str, tip_landmark, pip_landmark, mcp_landmark, threshold_factor=0.5):
        # (Same basic logic as before, logging shortened)
        if tip_landmark is None or pip_landmark is None or mcp_landmark is None: return False
        mcp_pip_dist_y = abs(mcp_landmark.y - pip_landmark.y); mcp_pip_dist_y = max(mcp_pip_dist_y, 0.001)
        tip_higher_than_pip = tip_landmark.y >= pip_landmark.y
        curl_threshold_y = pip_landmark.y - (mcp_pip_dist_y * threshold_factor)
        tip_close_below_pip = tip_landmark.y > curl_threshold_y
        is_curled = tip_higher_than_pip or tip_close_below_pip
        logging.debug(f"[{finger_name}_CURL_CHECK] -> Curled={is_curled}")
        return is_curled

    def _is_thumb_extended_out(self, thumb_tip, thumb_ip, thumb_mcp, index_mcp, y_threshold_factor=0.5, x_threshold_factor=1.2):
        # Renamed from _is_thumb_extended for clarity (used for Open Palm)
        # (Same logic as _is_thumb_extended before, logging shortened)
        if thumb_tip is None or thumb_ip is None or thumb_mcp is None or index_mcp is None: return False
        y_extension_threshold = thumb_ip.y - abs(thumb_ip.y - thumb_mcp.y) * y_threshold_factor
        is_extended_y = thumb_tip.y < y_extension_threshold
        thumb_base_dist_x = abs(thumb_mcp.x - index_mcp.x); thumb_base_dist_x = max(thumb_base_dist_x, 0.001)
        tip_dist_to_index_mcp_x = abs(thumb_tip.x - index_mcp.x)
        x_extension_threshold = thumb_base_dist_x * x_threshold_factor
        is_far_x = tip_dist_to_index_mcp_x > x_extension_threshold
        is_extended = is_extended_y and is_far_x
        logging.debug(f"[THUMB_EXT_OUT_CHECK] -> ExtendedOut={is_extended}")
        return is_extended

    # --- Robust Thumbs Up/Down Helpers ---

    def _check_thumb_direction(self, lm, num_curled_fingers):
        """
        Uses vector math to determine thumb direction relative to the hand.
        Returns 'up', 'down', or 'neutral'.
        Requires at least 3 other fingers to be curled.
        """
        if num_curled_fingers < 3: # Require most fingers curled
             return 'neutral'

        # Get points for vectors
        wrist = lm[HandLandmark.WRIST]
        thumb_ip = lm[HandLandmark.THUMB_IP]
        thumb_tip = lm[HandLandmark.THUMB_TIP]
        middle_mcp = lm[HandLandmark.MIDDLE_FINGER_MCP]

        if not all([wrist, thumb_ip, thumb_tip, middle_mcp]):
            logging.debug("[THUMB_DIR_CHECK] Missing landmarks for vector calculation.")
            return 'neutral'

        # Vector representing the approximate 'up' direction of the hand (wrist to middle base)
        hand_vector = self._get_vector(wrist, middle_mcp)
        hand_vector_normalized = self._normalize_vector(hand_vector)

        # Vector representing the thumb's pointing direction (IP to Tip)
        thumb_vector = self._get_vector(thumb_ip, thumb_tip)
        thumb_vector_normalized = self._normalize_vector(thumb_vector)

        if hand_vector_normalized is None or thumb_vector_normalized is None:
             logging.debug("[THUMB_DIR_CHECK] Could not normalize vectors.")
             return 'neutral'

        # Calculate dot product
        # Dot product > 0: Vectors point in generally similar directions (Thumb Down relative to vertical hand)
        # Dot product < 0: Vectors point in generally opposite directions (Thumb Up relative to vertical hand)
        # Dot product ~ 0: Vectors are roughly perpendicular
        dot_product = self._dot_product(thumb_vector_normalized, hand_vector_normalized)

        # Define thresholds for dot product (tune these!)
        # Larger threshold means thumb needs to point more clearly up/down.
        dot_threshold = 0.3 # Start with 0.4, adjust based on testing

        direction = 'neutral'
        if dot_product < -dot_threshold: # Pointing opposite to hand vector (Up)
            direction = 'up'
        elif dot_product > dot_threshold: # Pointing similar to hand vector (Down)
            direction = 'down'

        logging.debug(f"[THUMB_DIR_CHECK] DotProduct={dot_product:.3f} (Thresh={dot_threshold}), NumCurled={num_curled_fingers} -> Direction='{direction}'")
        return direction


    # --- Main Gesture Analysis Function ---

    def _analyze_hand_gesture(self, landmarks):
        """ Analyzes landmarks using vector math for Thumbs Up/Down. """
        if landmarks is None or len(landmarks) != 21: return "Analysis Error"
        try:
            lm = landmarks
            # --- Get Finger States ---
            index_extended = self._is_finger_extended("INDEX", lm[HandLandmark.INDEX_FINGER_TIP], lm[HandLandmark.INDEX_FINGER_PIP], lm[HandLandmark.INDEX_FINGER_MCP])
            middle_extended = self._is_finger_extended("MIDDLE", lm[HandLandmark.MIDDLE_FINGER_TIP], lm[HandLandmark.MIDDLE_FINGER_PIP], lm[HandLandmark.MIDDLE_FINGER_MCP])
            ring_extended = self._is_finger_extended("RING", lm[HandLandmark.RING_FINGER_TIP], lm[HandLandmark.RING_FINGER_PIP], lm[HandLandmark.RING_FINGER_MCP])
            pinky_extended = self._is_finger_extended("PINKY", lm[HandLandmark.PINKY_TIP], lm[HandLandmark.PINKY_PIP], lm[HandLandmark.PINKY_MCP])

            index_curled = self._is_finger_curled("INDEX", lm[HandLandmark.INDEX_FINGER_TIP], lm[HandLandmark.INDEX_FINGER_PIP], lm[HandLandmark.INDEX_FINGER_MCP])
            middle_curled = self._is_finger_curled("MIDDLE", lm[HandLandmark.MIDDLE_FINGER_TIP], lm[HandLandmark.MIDDLE_FINGER_PIP], lm[HandLandmark.MIDDLE_FINGER_MCP])
            ring_curled = self._is_finger_curled("RING", lm[HandLandmark.RING_FINGER_TIP], lm[HandLandmark.RING_FINGER_PIP], lm[HandLandmark.RING_FINGER_MCP])
            pinky_curled = self._is_finger_curled("PINKY", lm[HandLandmark.PINKY_TIP], lm[HandLandmark.PINKY_PIP], lm[HandLandmark.PINKY_MCP])

            thumb_extended_out = self._is_thumb_extended_out(lm[HandLandmark.THUMB_TIP], lm[HandLandmark.THUMB_IP], lm[HandLandmark.THUMB_MCP], lm[HandLandmark.INDEX_FINGER_MCP])

            num_main_fingers_curled = sum([index_curled, middle_curled, ring_curled, pinky_curled])

            # --- Classify Gestures (Order Matters!) ---

            # 1. Check Thumbs Up/Down (using vector math and relaxed curling)
            thumb_direction = self._check_thumb_direction(lm, num_main_fingers_curled)
            if thumb_direction == 'up':
                logging.debug("[Analyze] Determined: Thumbs Up")
                return "Thumbs Up"
            if thumb_direction == 'down':
                logging.debug("[Analyze] Determined: Thumbs Down")
                return "Thumbs Down"

            # 2. Check for OK Sign (requires extended fingers)
            # (Keep OK sign logic as tuned previously)
            thumb_index_dist = self._calculate_distance(lm[HandLandmark.THUMB_TIP], lm[HandLandmark.INDEX_FINGER_TIP])
            index_len = self._calculate_distance(lm[HandLandmark.INDEX_FINGER_TIP], lm[HandLandmark.INDEX_FINGER_MCP]); index_len = max(index_len, 0.01)
            ok_sign_threshold_factor = 0.7 # Assuming this was the tuned value
            ok_sign_threshold = index_len * ok_sign_threshold_factor
            is_thumb_index_close = thumb_index_dist < ok_sign_threshold
            logging.debug(f"[OK_SIGN_CHECK] Close={is_thumb_index_close}, MidExt={middle_extended}, RingExt={ring_extended}, PinkyExt={pinky_extended}")
            if is_thumb_index_close and middle_extended and ring_extended and pinky_extended:
                 logging.debug("[Analyze] Determined: OK Sign")
                 return "OK Sign"

            # 3. Check for Fist (requires *all* 4 fingers curled, fallback if thumb wasn't distinctly up/down)
            if num_main_fingers_curled == 4:
                 logging.debug("[Analyze] Determined: Fist")
                 return "Fist"

            # 4. Check for Open Palm (requires extended fingers AND thumb extended outwards)
            if index_extended and middle_extended and ring_extended and pinky_extended and thumb_extended_out:
                logging.debug("[Analyze] Determined: Open Palm")
                return "Open Palm"

            # --- Fallback ---
            logging.debug("[Analyze] No specific gesture matched -> Gesture Undefined")
            return "Gesture Undefined"

        except Exception as e:
            logging.error(f"[Analyze] Error: {e}", exc_info=True)
            return "Analysis Error"

    # recognize_gestures and __del__ remain unchanged
    def recognize_gestures(self, frame):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB); image.flags.writeable = False
        try: results = self.hands.process(image)
        except Exception as e: logging.error(f"MP Hands processing failed: {e}", exc_info=True); return []
        finally: image.flags.writeable = True
        processed_hands = []
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness_obj in zip(results.multi_hand_landmarks, results.multi_handedness):
                try:
                    handedness = handedness_obj.classification[0].label
                    gesture = self._analyze_hand_gesture(hand_landmarks.landmark) # Uses new logic
                    processed_hands.append({'landmarks': hand_landmarks, 'handedness': handedness, 'gesture': gesture})
                except Exception as e: logging.warning(f"Hand data processing failed: {e}", exc_info=False); continue
        return processed_hands

    def __del__(self):
        if hasattr(self, 'hands'): self.hands.close(); logging.info("MediaPipe Hands resources closed.")