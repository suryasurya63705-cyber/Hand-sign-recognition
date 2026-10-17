"""Hand tracking with the new MediaPipe Tasks API (works with new MediaPipe versions)."""
import os
import urllib.request
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_FILE = "hand_landmarker.task"
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")

CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
               (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
               (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]


class HandTracker:
    def __init__(self, confidence=0.7):
        if not os.path.exists(MODEL_FILE):
            print("Downloading hand model (one time, ~8 MB)...")
            urllib.request.urlretrieve(MODEL_URL, MODEL_FILE)
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODEL_FILE),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=confidence,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

    def detect(self, frame_bgr):
        """Returns (row of 63 numbers, landmarks) or (None, None) if no hand."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.detector.detect(img)
        if not result.hand_landmarks:
            return None, None
        lms = result.hand_landmarks[0]
        row = [v for p in lms for v in (p.x, p.y, p.z)]
        return row, lms

    @staticmethod
    def draw(frame, lms):
        h, w = frame.shape[:2]
        pts = [(int(p.x * w), int(p.y * h)) for p in lms]
        for a, b in CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)
        for x, y in pts:
            cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)
