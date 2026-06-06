import cv2
import numpy as np
import time
from datetime import datetime
from collections import deque, Counter

from model import EmotionPredictor
from tracker import CentroidTracker


predictor = EmotionPredictor()
tracker = CentroidTracker(max_distance=80)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

face_histories = {}
session_emotions = []

SMOOTHING_WINDOW = 10
PANEL_WIDTH = 380

EMOTION_ICONS = {
    "happy": ":)",
    "sad": ":(",
    "angry": ">:(",
    "surprise": ":O",
    "neutral": ":|"
}


def get_stable_emotion(face_id, emotion):
    if face_id not in face_histories:
        face_histories[face_id] = deque(maxlen=SMOOTHING_WINDOW)

    face_histories[face_id].append(emotion)
    return Counter(face_histories[face_id]).most_common(1)[0][0]


def get_overall_mood(emotions):
    if not emotions:
        return "No face"

    counts = Counter(emotions)
    top_emotion, _ = counts.most_common(1)[0]

    if len(counts) > 1:
        return f"Mostly {top_emotion}"

    return top_emotion


def get_message(mood):
    mood = mood.lower()

    if "happy" in mood:
        return "You seem cheerful today!"
    if "sad" in mood:
        return "Hope your day gets better."
    if "angry" in mood:
        return "Take a calm breath."
    if "surprise" in mood:
        return "Something surprised you?"
    if "neutral" in mood:
        return "Looking focused."

    return "Waiting for a face..."


def emotion_display(emotion):
    icon = EMOTION_ICONS.get(emotion.lower(), "")
    return f"{icon} {emotion}"


def save_session_report(start_time):
    end_time = datetime.now()
    duration = end_time - start_time

    with open("session_report.txt", "w", encoding="utf-8") as f:
        f.write("Emotion Mirror Pro - Session Report\n")
        f.write("==================================\n\n")
        f.write(f"Started At: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Ended At: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Duration: {str(duration).split('.')[0]}\n\n")

        if session_emotions:
            total = len(session_emotions)
            counts = Counter(session_emotions)

            f.write("Emotion Distribution:\n")

            for emotion, count in counts.most_common():
                percentage = (count / total) * 100
                f.write(f"{emotion}: {percentage:.2f}%\n")

            f.write(f"\nDominant Emotion: {counts.most_common(1)[0][0]}\n")
        else:
            f.write("No emotions detected during session.\n")


def draw_dashboard(frame, people_count, overall_mood, current_faces, fps):
    h, _, _ = frame.shape

    panel = np.zeros((h, PANEL_WIDTH, 3), dtype=np.uint8)
    panel[:] = (25, 25, 25)

    y = 35

    cv2.putText(
        panel,
        "Emotion Mirror Pro",
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    y += 35
    cv2.putText(
        panel,
        f"FPS: {fps:.1f}",
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (160, 220, 160),
        2
    )

    y += 40
    cv2.putText(
        panel,
        f"People Detected: {people_count}",
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (220, 220, 220),
        2
    )

    y += 35
    cv2.putText(
        panel,
        f"Overall: {overall_mood}",
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (220, 220, 220),
        2
    )

    y += 45
    cv2.putText(
        panel,
        "Current Faces",
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    y += 35

    if current_faces:
        for face_text in current_faces[:5]:
            cv2.putText(
                panel,
                face_text,
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (180, 180, 180),
                2
            )
            y += 30
    else:
        cv2.putText(
            panel,
            "No face visible",
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (180, 180, 180),
            2
        )
        y += 30

    y += 25
    cv2.putText(
        panel,
        "Session Summary",
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    y += 35

    if session_emotions:
        total = len(session_emotions)
        counts = Counter(session_emotions)

        for emotion, count in counts.most_common():
            percentage = (count / total) * 100
            label = f"{emotion}: {percentage:.1f}%"

            cv2.putText(
                panel,
                label,
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (180, 180, 180),
                2
            )

            bar_width = int((percentage / 100) * 160)

            cv2.rectangle(
                panel,
                (190, y - 15),
                (190 + bar_width, y - 5),
                (0, 180, 255),
                -1
            )

            y += 30
    else:
        cv2.putText(
            panel,
            "No emotions yet",
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (180, 180, 180),
            2
        )

    y += 35
    cv2.putText(
        panel,
        "Smart Message",
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    y += 35
    cv2.putText(
        panel,
        get_message(overall_mood),
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (180, 220, 180),
        2
    )

    y = h - 30
    cv2.putText(
        panel,
        "Press Q to quit",
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (160, 160, 160),
        2
    )

    return np.hstack((frame, panel))


cap = cv2.VideoCapture(0)

start_time = datetime.now()
prev_time = time.time()

while True:
    ret, frame = cap.read()

    if not ret:
        break

    current_time = time.time()
    fps = 1 / (current_time - prev_time) if current_time != prev_time else 0
    prev_time = current_time

    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    current_frame_emotions = []
    current_faces = []

    if len(faces) == 0:
        cv2.putText(
            frame,
            "No face detected",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

    tracked_faces = tracker.update(faces)

    for face_id, (_, _, box) in tracked_faces.items():
        x, y, w, h = box

        face_crop = frame[y:y + h, x:x + w]

        try:
            emotion, confidence = predictor.predict(face_crop)
            stable_emotion = get_stable_emotion(face_id, emotion)

            current_frame_emotions.append(stable_emotion)
            session_emotions.append(stable_emotion)

            display_emotion = emotion_display(stable_emotion)

            label = f"Face {face_id}: {display_emotion} {confidence * 100:.1f}%"
            current_faces.append(f"Face {face_id}: {display_emotion}")

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                label,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2
            )

        except Exception as e:
            print("Prediction error:", e)

    overall_mood = get_overall_mood(current_frame_emotions)

    output = draw_dashboard(
        frame,
        len(tracked_faces),
        overall_mood,
        current_faces,
        fps
    )

    cv2.imshow("Emotion Mirror Pro - v5", output)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()

save_session_report(start_time)

print("\n====== FINAL SESSION SUMMARY ======")

if session_emotions:
    total = len(session_emotions)
    counts = Counter(session_emotions)

    for emotion, count in counts.most_common():
        print(f"{emotion}: {(count / total) * 100:.2f}%")

    print("Dominant Emotion:", counts.most_common(1)[0][0])
    print("\nSession report saved as session_report.txt")
else:
    print("No emotions detected.")