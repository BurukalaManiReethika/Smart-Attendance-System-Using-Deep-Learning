"""
app.py - Smart Attendance System (Web Version)
------------------------------------------------
Flask backend that:
  - Serves a webcam page (browser camera via getUserMedia)
  - Receives frames from the browser, runs face recognition
  - Marks attendance (once per person per day) into a CSV
  - Lets you enroll new people via browser camera
  - Shows today's attendance list

Run locally:
    python app.py
Deploy on Render: see README_DEPLOY.md
"""

import os
import io
import base64
import pickle
import csv
import json
import smtplib
import threading
from email.mime.text import MIMEText
from datetime import datetime

import numpy as np
import cv2
import face_recognition
import requests
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWN_FACES_DIR = os.path.join(BASE_DIR, "known_faces")
ATTENDANCE_DIR = os.path.join(BASE_DIR, "attendance_records")
ENCODINGS_FILE = os.path.join(BASE_DIR, "encodings.pickle")
EMAILS_FILE = os.path.join(BASE_DIR, "emails.json")

TOLERANCE = 0.5
MODEL = "hog"

# --- Chatbot (Anthropic API) config ---
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
CHATBOT_ENABLED = bool(ANTHROPIC_API_KEY)

# --- Email (Gmail SMTP) config, read from environment variables ---
SMTP_EMAIL = os.environ.get("SMTP_EMAIL")        # your Gmail address, e.g. yourname@gmail.com
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD")  # Gmail App Password (NOT your normal password)
NOTIFY_ADMIN_EMAIL = os.environ.get("NOTIFY_ADMIN_EMAIL")  # optional: also notify this address on every mark
EMAIL_ENABLED = bool(SMTP_EMAIL and SMTP_PASSWORD)

os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
os.makedirs(ATTENDANCE_DIR, exist_ok=True)

# In-memory cache of encodings, loaded at startup / after enrollment
known_encodings = []
known_names = []


def load_encodings():
    global known_encodings, known_names
    if os.path.exists(ENCODINGS_FILE):
        with open(ENCODINGS_FILE, "rb") as f:
            data = pickle.load(f)
        known_encodings = data.get("encodings", [])
        known_names = data.get("names", [])
    else:
        known_encodings, known_names = [], []


def save_encodings():
    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump({"encodings": known_encodings, "names": known_names}, f)


def load_emails():
    if os.path.exists(EMAILS_FILE):
        with open(EMAILS_FILE, "r") as f:
            return json.load(f)
    return {}


def save_email_for_person(name, email):
    emails = load_emails()
    if email:
        emails[name] = email
        with open(EMAILS_FILE, "w") as f:
            json.dump(emails, f, indent=2)


def send_email_async(to_address, subject, body):
    """Send an email in a background thread so it never blocks/slow the request."""
    if not EMAIL_ENABLED or not to_address:
        return

    def _send():
        try:
            msg = MIMEText(body)
            msg["Subject"] = subject
            msg["From"] = SMTP_EMAIL
            msg["To"] = to_address
            with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=15) as server:
                server.login(SMTP_EMAIL, SMTP_PASSWORD)
                server.sendmail(SMTP_EMAIL, [to_address], msg.as_string())
            print(f"[EMAIL SENT] Successfully sent to {to_address}")
        except Exception as e:
            print(f"[EMAIL ERROR] Could not send to {to_address}: {type(e).__name__}: {e}")

    threading.Thread(target=_send, daemon=True).start()


def notify_attendance_marked(name):
    """Send confirmation email to the person, and optionally to an admin address."""
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    emails = load_emails()
    person_email = emails.get(name)

    print(f"[EMAIL DEBUG] notify_attendance_marked called for '{name}'. "
          f"EMAIL_ENABLED={EMAIL_ENABLED}, saved_emails={emails}, person_email={person_email}")

    if not EMAIL_ENABLED:
        print("[EMAIL SKIP] SMTP_EMAIL / SMTP_PASSWORD not set in environment.")
        return

    if person_email:
        send_email_async(
            person_email,
            "Attendance Marked — Smart Attendance System",
            f"Hi {name},\n\nYour attendance was marked present on {now_str}.\n\n"
            f"— Smart Attendance System"
        )
    else:
        print(f"[EMAIL SKIP] No email saved for '{name}'. Available names: {list(emails.keys())}")

    if NOTIFY_ADMIN_EMAIL:
        send_email_async(
            NOTIFY_ADMIN_EMAIL,
            f"Attendance: {name} marked present",
            f"{name} was marked present at {now_str}."
        )


def get_all_attendance_records():
    """Read every attendance CSV file and return a combined text summary."""
    all_records = []
    if os.path.isdir(ATTENDANCE_DIR):
        for filename in sorted(os.listdir(ATTENDANCE_DIR)):
            if not filename.endswith(".csv"):
                continue
            filepath = os.path.join(ATTENDANCE_DIR, filename)
            with open(filepath, "r", newline="") as f:
                reader = csv.reader(f)
                next(reader, None)
                for row in reader:
                    if row:
                        all_records.append(f"{row[0]} — {row[1]} at {row[2]}")
    return all_records


def ask_chatbot(question):
    """Send the user's question + attendance context to Claude and return the answer."""
    if not CHATBOT_ENABLED:
        return "Chatbot is not configured yet. Set the ANTHROPIC_API_KEY environment variable to enable it."

    records = get_all_attendance_records()
    enrolled_people = sorted(set(known_names))
    today_marked = sorted(load_already_marked())

    context = (
        f"Today's date: {datetime.now().strftime('%Y-%m-%d')}\n"
        f"Enrolled people in the system: {', '.join(enrolled_people) if enrolled_people else 'None enrolled yet'}\n"
        f"People marked present today: {', '.join(today_marked) if today_marked else 'None yet'}\n\n"
        f"Full attendance history (name — date at time):\n"
        + ("\n".join(records) if records else "No attendance records yet.")
    )

    system_prompt = (
        "You are an attendance assistant for a Smart Attendance System. "
        "Answer the user's question using ONLY the attendance data provided below. "
        "Be concise and direct. If the data doesn't answer the question, say so clearly.\n\n"
        f"ATTENDANCE DATA:\n{context}"
    )

    try:
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-sonnet-5",
                "max_tokens": 400,
                "system": system_prompt,
                "messages": [{"role": "user", "content": question}],
            },
            timeout=30,
        )
        if not response.ok:
            print(f"[CHATBOT ERROR] {response.status_code}: {response.text}")
        response.raise_for_status()
        data = response.json()
        text_parts = [block["text"] for block in data.get("content", []) if block.get("type") == "text"]
        return "\n".join(text_parts) if text_parts else "I couldn't generate a response. Please try again."
    except Exception as e:
        print(f"[CHATBOT ERROR] {type(e).__name__}: {e}")
        return "Sorry, I couldn't reach the assistant right now. Please try again in a moment."


def rebuild_encodings_from_disk():
    """Re-scan known_faces/ and rebuild encodings.pickle from scratch."""
    global known_encodings, known_names
    new_encodings, new_names = [], []

    for person_name in os.listdir(KNOWN_FACES_DIR):
        person_dir = os.path.join(KNOWN_FACES_DIR, person_name)
        if not os.path.isdir(person_dir):
            continue
        for image_file in os.listdir(person_dir):
            if not image_file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            image_path = os.path.join(person_dir, image_file)
            image = face_recognition.load_image_file(image_path)
            boxes = face_recognition.face_locations(image, model=MODEL)
            encs = face_recognition.face_encodings(image, boxes)
            if encs:
                new_encodings.append(encs[0])
                new_names.append(person_name)

    known_encodings, known_names = new_encodings, new_names
    save_encodings()


def decode_base64_image(data_url):
    """Convert a 'data:image/jpeg;base64,...' string into an OpenCV BGR image."""
    header, encoded = data_url.split(",", 1)
    img_bytes = base64.b64decode(encoded)
    np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return frame


def get_today_csv_path():
    today = datetime.now().strftime("%Y-%m-%d")
    return os.path.join(ATTENDANCE_DIR, f"attendance_{today}.csv")


def load_already_marked():
    csv_path = get_today_csv_path()
    marked = set()
    if os.path.exists(csv_path):
        with open(csv_path, "r", newline="") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if row:
                    marked.add(row[0])
    return marked


def mark_attendance(name):
    csv_path = get_today_csv_path()
    marked = load_already_marked()
    if name in marked:
        return False
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Name", "Date", "Time"])
        now = datetime.now()
        writer.writerow([name, now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")])
    return True


# ---------------------------------------------------------------- Routes

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/enroll")
def enroll_page():
    return render_template("enroll.html")


@app.route("/attendance")
def attendance_page():
    csv_path = get_today_csv_path()
    rows = []
    if os.path.exists(csv_path):
        with open(csv_path, "r", newline="") as f:
            reader = csv.reader(f)
            next(reader, None)
            rows = list(reader)
    return render_template("attendance.html", rows=rows, today=datetime.now().strftime("%Y-%m-%d"))


@app.route("/api/recognize", methods=["POST"])
def api_recognize():
    """Receive one frame from the browser, recognize faces, mark attendance."""
    payload = request.get_json(force=True)
    image_data = payload.get("image")
    if not image_data:
        return jsonify({"error": "No image provided"}), 400

    frame = decode_base64_image(image_data)
    if frame is None:
        return jsonify({"error": "Could not decode image"}), 400

    small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small, model=MODEL)
    face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

    results = []
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        name = "Unknown"
        marked_now = False

        if known_encodings:
            distances = face_recognition.face_distance(known_encodings, face_encoding)
            best_idx = int(distances.argmin())
            if distances[best_idx] <= TOLERANCE:
                name = known_names[best_idx]
                marked_now = mark_attendance(name)
                if marked_now:
                    notify_attendance_marked(name)

        # scale box coords back up (we resized by 0.5)
        results.append({
            "name": name,
            "box": [int(top * 2), int(right * 2), int(bottom * 2), int(left * 2)],
            "marked_now": marked_now
        })

    return jsonify({"faces": results, "marked_today": sorted(load_already_marked())})


@app.route("/api/enroll", methods=["POST"])
def api_enroll():
    """Receive a name + list of base64 images, save them, rebuild encodings."""
    payload = request.get_json(force=True)
    name = payload.get("name", "").strip().replace(" ", "_")
    email = payload.get("email", "").strip()
    images = payload.get("images", [])

    if not name:
        return jsonify({"error": "Name is required"}), 400
    if not images:
        return jsonify({"error": "No images provided"}), 400

    person_dir = os.path.join(KNOWN_FACES_DIR, name)
    os.makedirs(person_dir, exist_ok=True)

    saved = 0
    for i, img_data in enumerate(images):
        frame = decode_base64_image(img_data)
        if frame is None:
            continue
        # quick check: does this photo actually contain a face?
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb, model=MODEL)
        if not boxes:
            continue
        filename = os.path.join(person_dir, f"{name}_{i}.jpg")
        cv2.imwrite(filename, frame)
        saved += 1

    if saved == 0:
        return jsonify({"error": "No faces detected in captured photos. Try again with better lighting."}), 400

    save_email_for_person(name, email)
    rebuild_encodings_from_disk()

    return jsonify({"message": f"Enrolled '{name}' with {saved} photo(s).", "total_people": len(set(known_names))})


@app.route("/chat")
def chat_page():
    return render_template("chat.html")


@app.route("/api/chat", methods=["POST"])
def api_chat():
    payload = request.get_json(force=True)
    question = payload.get("question", "").strip()
    if not question:
        return jsonify({"error": "Question is required"}), 400

    answer = ask_chatbot(question)
    return jsonify({"answer": answer})


@app.route("/api/people", methods=["GET"])
def api_people():
    return jsonify({"people": sorted(set(known_names))})


load_encodings()
print(f"[STARTUP] EMAIL_ENABLED={EMAIL_ENABLED} | SMTP_EMAIL={'SET' if SMTP_EMAIL else 'NOT SET'} | "
      f"SMTP_PASSWORD={'SET' if SMTP_PASSWORD else 'NOT SET'} | CHATBOT_ENABLED={CHATBOT_ENABLED} | "
      f"known_people={sorted(set(known_names))}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
