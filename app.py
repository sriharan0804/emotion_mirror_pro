import cv2
import numpy as np
from collections import deque, Counter
from model import EmotionPredictor


predictor = EmotionPredictor()

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

face_histories = {}
session_emotions = []

SMOOTHING_WINDOW = 10
PANEL_WIDTH = 360


def get_stable_emotion(face_id, emotion):
    if face_id not in face_histories:
        face_histories[face_id] = deque(maxlen=SMOOTHING_WINDOW)

    face_histories[face_id].append(emotion)
    return Counter(face_histories[face_id]).most_common(1)[0][0]


def get_overall_mood(emotions):
    if not emotions:
        return "No face"

    counts = Counter(emotions)
    top_emotion, top_count = counts.most_common(1)[0]

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


def draw_dashboard(frame, people_count, overall_mood):
    h, w, _ = frame.shape

    panel = np.zeros((h, PANEL_WIDTH, 3), dtype=np.uint8)
    panel[:] = (25, 25, 25)

    y = 40

    cv2.putText(panel, "Emotion Mirror Pro", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    y += 50
    cv2.putText(panel, f"People: {people_count}", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    y += 40
    cv2.putText(panel, f"Mood: {overall_mood}", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2)

    y += 50
    cv2.putText(panel, "Session Summary", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    y += 35

    if session_emotions:
        total = len(session_emotions)
        counts = Counter(session_emotions)

        for emotion, count in counts.most_common():
            percentage = (count / total) * 100

            cv2.putText(panel, f"{emotion}: {percentage:.1f}%", (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 2)

            bar_width = int((percentage / 100) * 180)
            cv2.rectangle(panel, (150, y - 15), (150 + bar_width, y - 5),
                          (0, 180, 255), -1)

            y += 35
    else:
        cv2.putText(panel, "No emotions yet", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 2)

    y += 30
    cv2.putText(panel, "Smart Message", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    y += 35
    message = get_message(overall_mood)

    cv2.putText(panel, message, (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 220, 180), 2)

    y = h - 30
    cv2.putText(panel, "Press Q to quit", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 2)

    combined = np.hstack((frame, panel))
    return combined


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    current_frame_emotions = []

    if len(faces) == 0:
        cv2.putText(frame, "No face detected", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    for i, (x, y, w, h) in enumerate(faces):
        face_crop = frame[y:y+h, x:x+w]

        try:
            emotion, confidence = predictor.predict(face_crop)
            stable_emotion = get_stable_emotion(i, emotion)

            current_frame_emotions.append(stable_emotion)
            session_emotions.append(stable_emotion)

            label = f"Face {i+1}: {stable_emotion} {confidence*100:.1f}%"

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        except Exception as e:
            print("Prediction error:", e)

    overall_mood = get_overall_mood(current_frame_emotions)
    output = draw_dashboard(frame, len(faces), overall_mood)

    cv2.imshow("Emotion Mirror Pro - v3 Dashboard", output)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()

print("\n====== FINAL SESSION SUMMARY ======")

if session_emotions:
    total = len(session_emotions)
    counts = Counter(session_emotions)

    for emotion, count in counts.most_common():
        print(f"{emotion}: {(count / total) * 100:.2f}%")

    print("Dominant Emotion:", counts.most_common(1)[0][0])
else:
    print("No emotions detected.")