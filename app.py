import cv2
from collections import deque, Counter
from model import EmotionPredictor


predictor = EmotionPredictor()

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Stores recent predictions for each face index
face_histories = {}

# Overall session emotion counts
session_emotions = []

SMOOTHING_WINDOW = 10


def get_stable_emotion(face_id, emotion):
    if face_id not in face_histories:
        face_histories[face_id] = deque(maxlen=SMOOTHING_WINDOW)

    face_histories[face_id].append(emotion)

    emotion_counts = Counter(face_histories[face_id])
    stable_emotion = emotion_counts.most_common(1)[0][0]

    return stable_emotion


def get_overall_mood(emotions):
    if not emotions:
        return "No face"

    emotion_counts = Counter(emotions)
    top_emotion, top_count = emotion_counts.most_common(1)[0]

    if len(emotion_counts) > 1:
        return f"Mixed / Mostly {top_emotion}"

    return top_emotion


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    current_frame_emotions = []

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

    for i, (x, y, w, h) in enumerate(faces):
        face_crop = frame[y:y+h, x:x+w]

        try:
            emotion, confidence = predictor.predict(face_crop)

            stable_emotion = get_stable_emotion(i, emotion)
            current_frame_emotions.append(stable_emotion)
            session_emotions.append(stable_emotion)

            label = f"Face {i+1}: {stable_emotion} ({confidence*100:.1f}%)"

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            cv2.putText(
                frame,
                label,
                (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        except Exception as e:
            print("Prediction error:", e)

    overall_mood = get_overall_mood(current_frame_emotions)

    cv2.rectangle(frame, (0, 0), (frame.shape[1], 80), (30, 30, 30), -1)

    cv2.putText(
        frame,
        f"People Detected: {len(faces)}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Overall Mood: {overall_mood}",
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.imshow("Emotion Mirror Pro", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()


print("\n====== SESSION SUMMARY ======")

if session_emotions:
    total = len(session_emotions)
    counts = Counter(session_emotions)

    for emotion, count in counts.most_common():
        percentage = (count / total) * 100
        print(f"{emotion}: {percentage:.2f}%")

    dominant_emotion = counts.most_common(1)[0][0]
    print(f"\nDominant Session Emotion: {dominant_emotion}")
else:
    print("No emotions detected during session.")