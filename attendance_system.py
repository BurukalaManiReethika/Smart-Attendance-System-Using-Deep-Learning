"""
attendance_system.py
---------------------
Real-time face-recognition attendance system.

- Opens webcam feed
- Detects & recognizes faces using encodings.pickle (built by encode_faces.py)
- Marks attendance (name, date, time) once per person per day
- Saves attendance to attendance_records/attendance_<date>.csv

Usage:
    python attendance_system.py

Controls:
    q  -> quit
"""

import cv2
import face_recognition
import pickle
import os
import csv
from datetime import datetime

ENCODINGS_FILE = "encodings.pickle"
ATTENDANCE_DIR = "attendance_records"
CAM_INDEX = 0
TOLERANCE = 0.5          # lower = stricter match (0.4-0.6 typical range)
MODEL = "hog"            # "hog" (fast, CPU) or "cnn" (accurate, needs GPU)
RESIZE_SCALE = 0.25      # shrink frame for faster detection (0.25 = quarter size)


def load_encodings():
    if not os.path.exists(ENCODINGS_FILE):
        print(f"'{ENCODINGS_FILE}' not found. Run enroll_faces.py then encode_faces.py first.")
        exit(1)
    with open(ENCODINGS_FILE, "rb") as f:
        return pickle.load(f)


def get_today_csv_path():
    os.makedirs(ATTENDANCE_DIR, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")
    return os.path.join(ATTENDANCE_DIR, f"attendance_{today}.csv")


def load_already_marked(csv_path):
    marked = set()
    if os.path.exists(csv_path):
        with open(csv_path, "r", newline="") as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header
            for row in reader:
                if row:
                    marked.add(row[0])
    return marked


def mark_attendance(name, csv_path, marked_set):
    if name in marked_set or name == "Unknown":
        return False

    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Name", "Date", "Time"])
        now = datetime.now()
        writer.writerow([name, now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")])

    marked_set.add(name)
    print(f"[ATTENDANCE MARKED] {name} at {datetime.now().strftime('%H:%M:%S')}")
    return True


def run():
    data = load_encodings()
    known_encodings = data["encodings"]
    known_names = data["names"]

    csv_path = get_today_csv_path()
    marked_today = load_already_marked(csv_path)
    print(f"Attendance file for today: {csv_path}")
    print(f"Already marked today: {marked_today if marked_today else 'None'}\n")

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("Could not open webcam. Check CAM_INDEX or camera permissions.")
        return

    print("Starting attendance system... Press 'q' to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=RESIZE_SCALE, fy=RESIZE_SCALE)
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small, model=MODEL)
        face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=TOLERANCE)
            distances = face_recognition.face_distance(known_encodings, face_encoding)

            name = "Unknown"
            if len(distances) > 0:
                best_match_idx = distances.argmin()
                if matches[best_match_idx]:
                    name = known_names[best_match_idx]
                    mark_attendance(name, csv_path, marked_today)

            # Scale coordinates back up to original frame size
            top = int(top / RESIZE_SCALE)
            right = int(right / RESIZE_SCALE)
            bottom = int(bottom / RESIZE_SCALE)
            left = int(left / RESIZE_SCALE)

            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            label = f"{name} - PRESENT" if name != "Unknown" else "Unknown"
            cv2.putText(frame, label, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.putText(frame, f"Marked today: {len(marked_today)}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.imshow("Smart Attendance System", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nSession ended. Total marked today: {len(marked_today)} -> {sorted(marked_today)}")


if __name__ == "__main__":
    run()
