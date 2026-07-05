"""
encode_faces.py
----------------
Scans the known_faces/ directory (one sub-folder per person, containing
their photos), computes a 128-d face encoding for each photo, and stores
all encodings + names in encodings.pickle for use by attendance_system.py.

Usage:
    python encode_faces.py
"""

import face_recognition
import os
import pickle

KNOWN_FACES_DIR = "known_faces"
ENCODINGS_FILE = "encodings.pickle"
MODEL = "hog"  # "hog" = faster (CPU), "cnn" = more accurate (needs GPU ideally)


def build_encodings():
    known_encodings = []
    known_names = []

    if not os.path.isdir(KNOWN_FACES_DIR):
        print(f"Folder '{KNOWN_FACES_DIR}' not found. Run enroll_faces.py first.")
        return

    people = [p for p in os.listdir(KNOWN_FACES_DIR)
              if os.path.isdir(os.path.join(KNOWN_FACES_DIR, p))]

    if not people:
        print("No enrolled people found. Run enroll_faces.py first.")
        return

    for person_name in people:
        person_dir = os.path.join(KNOWN_FACES_DIR, person_name)
        image_files = [f for f in os.listdir(person_dir)
                       if f.lower().endswith((".jpg", ".jpeg", ".png"))]

        print(f"Processing '{person_name}' ({len(image_files)} images)...")

        for image_file in image_files:
            image_path = os.path.join(person_dir, image_file)
            image = face_recognition.load_image_file(image_path)

            boxes = face_recognition.face_locations(image, model=MODEL)
            encodings = face_recognition.face_encodings(image, boxes)

            if len(encodings) == 0:
                print(f"  [!] No face found in {image_file}, skipping.")
                continue

            # If multiple faces detected in one photo, use the first one
            known_encodings.append(encodings[0])
            known_names.append(person_name)

    if not known_encodings:
        print("No usable face encodings were generated. Check your photos.")
        return

    data = {"encodings": known_encodings, "names": known_names}
    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump(data, f)

    print(f"\nDone! {len(known_encodings)} encodings saved to '{ENCODINGS_FILE}'.")
    print(f"Enrolled people: {sorted(set(known_names))}")


if __name__ == "__main__":
    build_encodings()
