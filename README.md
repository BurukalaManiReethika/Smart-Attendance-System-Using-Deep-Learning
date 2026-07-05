# Smart Attendance System (Face Recognition)

A working Python project that marks attendance automatically using your webcam and face recognition.

## Folder Structure
```
smart_attendance/
├── known_faces/            # photos of enrolled people (auto-created)
├── attendance_records/     # daily attendance CSV files (auto-created)
├── enroll_faces.py         # Step 1: capture photos of a new person
├── encode_faces.py         # Step 2: build face encodings from photos
├── attendance_system.py    # Step 3: run live attendance marking
└── requirements.txt
```

## Setup

### 1. Install Python
Python 3.9 – 3.11 recommended (dlib install is easiest on these versions).

### 2. Install dependencies

**Windows:** `dlib` needs CMake + Visual Studio Build Tools. Easiest path:
```bash
pip install cmake
pip install dlib
pip install -r requirements.txt
```
If `dlib` fails to build, install "Desktop development with C++" from Visual Studio Build Tools first, then retry.

**Mac:**
```bash
brew install cmake
pip install -r requirements.txt
```

**Linux:**
```bash
sudo apt-get install build-essential cmake
pip install -r requirements.txt
```

## How to Use

### Step 1 — Enroll people
Run this once per person you want the system to recognize:
```bash
python enroll_faces.py
```
- Enter their name (e.g. `Ravi_Kumar`)
- Look at the camera, press **SPACE** 5 times to capture 5 photos (different angles: front, slight left/right, smiling, neutral)
- Press **ESC** anytime to stop early

Repeat for every person you want enrolled.

### Step 2 — Generate encodings
After enrolling everyone:
```bash
python encode_faces.py
```
This reads all photos in `known_faces/` and creates `encodings.pickle` (the "database" of known faces).

Re-run this any time you add/enroll a new person.

### Step 3 — Run attendance system
```bash
python attendance_system.py
```
- Opens your webcam
- Green box + name = recognized, attendance marked automatically
- Red box + "Unknown" = face not recognized
- Each person is marked only **once per day**
- Press **q** to quit

Attendance gets saved to `attendance_records/attendance_YYYY-MM-DD.csv` with columns: `Name, Date, Time`.

## Tuning Tips
- `TOLERANCE` in `attendance_system.py` (default `0.5`): lower it (e.g. `0.4`) for stricter matching if you get false positives; raise it (e.g. `0.6`) if real people aren't being recognized.
- `MODEL = "hog"` is fast and works fine on CPU/laptop. Switch to `"cnn"` only if you have a GPU — it's more accurate but much slower on CPU.
- Enroll each person with 5-10 varied photos (different lighting/angle) for better accuracy.

## How It Works (Deep Learning Part)
- `face_recognition` library uses a pre-trained **ResNet-based CNN (dlib)** to detect faces and convert each face into a **128-dimension embedding vector**.
- Two photos of the same person produce embeddings that are close together (small Euclidean distance); different people produce embeddings far apart.
- Recognition = comparing the live face's embedding against all stored embeddings and picking the closest match below the tolerance threshold.

## Possible Next Steps
- Add a Streamlit/Flask web dashboard to view attendance reports
- Add liveness/anti-spoofing detection (prevent marking attendance from a photo held up to the camera)
- Export monthly attendance summary to Excel automatically
- Send absentee alerts via email/SMS
