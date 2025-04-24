# camera.py (Simplified for Web App)
import cv2
import logging
import time
# No mediapipe drawing needed here anymore

class Camera:
    """Handles camera initialization and frame capture."""
    def __init__(self, camera_index=0): # Removed window_name
        self.camera_index = camera_index
        self.cap = None
        logging.info(f"Camera class initialized for index {camera_index}.")

    def start_camera(self):
        """Initializes the camera capture."""
        logging.info(f"Attempting to start camera at index {self.camera_index}...")
        # Using CAP_DSHOW can sometimes help on Windows
        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            logging.error(f"CRITICAL: Cannot open camera with index {self.camera_index}.")
            raise IOError(f"Cannot open camera {self.camera_index}")
        logging.info(f"Camera {self.camera_index} opened successfully.")

    def capture_frame(self):
        """Captures a single frame from the camera."""
        if not self.cap or not self.cap.isOpened():
            return None # Return None if camera isn't ready

        ret, frame = self.cap.read()
        if not ret:
            logging.warning("Error: Cannot read frame from camera stream.")
            return None # Return None if reading fails
        return frame

    def release(self):
        """Releases the camera resource."""
        if self.cap:
            self.cap.release()
            logging.info("Camera resource released.")
            self.cap = None

    def is_opened(self):
        """Check if the camera is open."""
        return self.cap is not None and self.cap.isOpened()