import cv2
import time
import threading
import joblib
import numpy as np
import pyttsx3
from collections import deque, Counter
from utils import normalize
from hand_tracker import HandTracker

model = joblib.load("model.pkl")
tracker = HandTracker()


def speak(text):
    def run():
        try:
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
        except Exception:
            pass
    threading.Thread(target=run, daemon=True).start()   # thread so the video doesn't freeze


recent = deque(maxlen=10)       # last 10 predictions for smoothing
sentence = []
hold_sign, hold_start, added = None, 0, False
shown, conf = "-", 0.0

cap = cv2.VideoCapture(0)
while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    row, lms = tracker.detect(frame)

    if row is not None:
        tracker.draw(frame, lms)
        probs = model.predict_proba([normalize(row)])[0]
        i = int(np.argmax(probs))
        sign, conf = model.classes_[i], probs[i]
        recent.append(sign if conf >= 0.7 else "?")

        stable, votes = Counter(recent).most_common(1)[0]   # majority vote
        if votes < 7:
            stable = "?"
        shown = stable

        if stable != "?":
            if stable != hold_sign:
                hold_sign, hold_start, added = stable, time.time(), False
            elif not added and time.time() - hold_start >= 1.0:   # held for 1 second
                sentence.append(stable.replace("_", " ").upper())
                speak(stable.replace("_", " "))
                added = True
        else:
            hold_sign = None
    else:
        recent.clear()
        hold_sign, shown = None, "-"

    cv2.putText(frame, f"Sign: {shown.upper()} ({conf*100:.0f}%)", (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.putText(frame, "Sentence: " + " ".join(sentence[-6:]), (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.putText(frame, "C = clear   Q = quit", (10, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.imshow("Sign Recognition", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("c"):
        sentence.clear()

cap.release()
cv2.destroyAllWindows()
