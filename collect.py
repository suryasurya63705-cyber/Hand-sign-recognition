import cv2
import csv
import os
from hand_tracker import HandTracker

FILE = input("File name (data.csv for you, friend.csv for your friend): ").strip() or "data.csv"
label = input("Sign name (e.g. hello, yes, no, peace, one, three, L, Y, ok, C): ").strip()

tracker = HandTracker()

# create the CSV with a header the first time
if not os.path.exists(FILE):
    with open(FILE, "w", newline="") as f:
        header = [f"{a}{i}" for i in range(21) for a in "xyz"] + ["label"]
        csv.writer(f).writerow(header)

cap = cv2.VideoCapture(0)
recording = False
count = 0

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    row, lms = tracker.detect(frame)

    if row is not None:
        tracker.draw(frame, lms)
        if recording:
            with open(FILE, "a", newline="") as f:
                csv.writer(f).writerow(row + [label])
            count += 1

    status = "RECORDING" if recording else "PAUSED"
    cv2.putText(frame, f"{label} | {status} | samples: {count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, "SPACE = start/stop   Q = quit", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.imshow("Collect", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord(" "):
        recording = not recording
    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print(f"Saved {count} samples of '{label}' to {FILE}")
