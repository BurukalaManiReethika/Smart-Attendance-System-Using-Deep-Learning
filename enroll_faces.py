"""
enroll_faces.py
----------------
Capture photos of a new person using the webcam and save them under
known_faces/<person_name>/ so encode_faces.py can process them later.

Usage:
    python enroll_faces.py
"""

import cv2
import os

KNOWN_FACES_DIR = "known_faces"
NUM_PHOTOS = 5          # how many photos to capture per person
CAM_INDEX = 0           # change if you have multiple cameras


def enroll_person():
    name = input("Enter person's name (no spaces, e.g. John_Doe): ").strip()
    if not name:
        print("Name cannot be empty.")
        return

    person_dir = os.path.join(KNOWN_FACES_DIR, name)
    os.makedirs(person_dir, exist_ok=True)

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("Could not open webcam. Check CAM_INDEX or camera permissions.")
        return

    print(f"\nCapturing {NUM_PHOTOS} photos for '{name}'.")
    print("Press SPACE to capture a photo, ESC to quit early.\n")

    count = 0
    while count < NUM_PHOTOS:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from camera.")
            break

        display = frame.copy()
        cv2.putText(display, f"Photos captured: {count}/{NUM_PHOTOS}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(display, "SPACE = capture | ESC = quit",
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow("Enroll Face - " + name, display)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == 32:  # SPACE
            count += 1
            filepath = os.path.join(person_dir, f"{name}_{count}.jpg")
            cv2.imwrite(filepath, frame)
            print(f"Saved: {filepath}")

    cap.release()
    cv2.destroyAllWindows()

    if count > 0:
        print(f"\nDone! {count} photo(s) saved for '{name}'.")
        print("Now run: python encode_faces.py")
    else:
        print("No photos captured.")


if __name__ == "__main__":
    enroll_person()
