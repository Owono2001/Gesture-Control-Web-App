# test_install.py
import numpy as np
import cv2
import tensorflow as tf
import mediapipe as mp
import sys

print(f"Python Version: {sys.version}")
print(f"NumPy Version: {np.__version__}")
print(f"OpenCV Version: {cv2.__version__}")
print(f"TensorFlow Version: {tf.__version__}")
print(f"MediaPipe Version: {mp.__version__}")

print("\nSuccessfully imported all libraries!")