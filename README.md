-----Emotion Mirror Pro-----

A real-time emotion recognition application that uses a webcam feed to detect
faces and predict emotions live on screen. The application supports multiple
people in a frame, emotion stabilization, mood analysis, session statistics, 
and a live dashboard for a more polished user experience.

________________________________________________________________________________
-----Features-----
-> Core Features:
  * Real-time webcam face detection
  * Real-time emotion recognition
  * Multiple-face emotion detection
  * Confidence score display
  * Support for Happy, Sad, Angry, Surprise, and Neutral emotions
    
-> Enhanced Features:
  * Stable emotion prediction using temporal smoothing
  * Overall mood detection across all visible faces
  * Live dashboard panel
  * Face tracking with persistent face IDs
  * Session emotion statistics
  * Automatic session report generation
  * FPS monitoring
  * Smart mood-based messages
  * No-face handling

_________________________________________________________________________________
-----Demo-----
-> Single Face Detection:

-> Multiple Face Detection:

-> Dashboard View:

-> Session Report:

___________________________________________________________________________________
-----Project Structure-----
```text
emotion_mirror_pro/
│
├── app.py                    # Main application
├── model.py                  # CNN model and emotion predictor
├── tracker.py                # Centroid-based face tracking
├── Dataset.py                # Dataset utilities
├── train_model.py            # Model training script
├── requirement.txt           # Project dependencies
├── README.md                 # Project documentation
├── session_report.txt        # Generated session report
│
├── emotion_cnn_best.pth      # Trained model weights
│
├── emotion_dataset/
│   ├── train/
│   │   ├── angry/
│   │   ├── disgust/
│   │   ├── fear/
│   │   ├── happy/
│   │   ├── neutral/
│   │   ├── sad/
│   │   └── surprise/
│   │
│   └── test/
│       ├── angry/
│       ├── disgust/
│       ├── fear/
│       ├── happy/
│       ├── neutral/
│       ├── sad/
│       └── surprise/
│
└── screenshots/
    ├── single_face.png
    ├── multiple_faces.png
    ├── dashboard.png
    └── session_report.png
```
_____________________________________________________________________________________
-----Tech Stack-----
  * Python
  * Pytorch
  * OpenCV
  * NumPy
  * TorchVision

_____________________________________________________________________________________
-----Model Details-----
The emotion classifier is a custom Convolutional Neural Network (CNN) trained
on a facial expression dataset containing the following classes:
  * Angry
  * Happy
  * Neutral
  * Sad
  * Suprise
  * Disgust
  * Fear
During infrence:
  * Disgust is merged into Angry
  * Fear is merged into Suprise
This aligns the predictions with the five emotions requested in the assignment.

______________________________________________________________________________________
-----Model Performance-----
