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

  * Training Framework: PyTorch
  * Architecture: Custom CNN
  * Test Accuracy: ~65%
_______________________________________________________________________________________
-----installation-----

-> Clone the repository:
git clone <YOUR_GITHUB_REPO_URL> 
cd emotion_mirror_pro

-> Create a virtual environment:
python -m venv venv

-> Activate the environment:
Windows:
venv\Scripts\activate

-> Install dependencies:
pip install -r requirements.txt

_________________________________________________________________________________________
-----Running the Application-----

Make sure the trained model exists at:
  emotion_cnn_best.pth
Run:
  python app.py
Press:
  Q 
 to quit the application

 _________________________________________________________________________________________
 -----How it Works-----
 * Webcam frames are captured using OpenCV.
 * Faces are detected using Haar Cascade face detection.
 * Each face is tracked using a centroid-based tracker.
 * The trained CNN predicts emotions for each face.
 * Predictions are stabilized using temporal smoothing.
 * Dashboard statistics are updated in real time.
 * Session statistics are saved automatically when the application closes.

 __________________________________________________________________________________________
 -----Challenge Faced-----

 One challenge was handling unstable emotion predictions between consecutive frames. To 
 improve the user experience, a temporal smoothing mechanism was introduced that considers 
 recent predictions before displaying the final emotion.

Another challenge was maintaining consistent identities when multiple faces were present in 
the frame. A centroid-based tracking system was implemented so that each detected person 
retains a stable face ID throughout the session.

____________________________________________________________________________________________
-----What I Learned-----

Through this project I gained hands-on experience with:

  * Real-time computer vision applications
  * CNN-based emotion recognition
  * PyTorch model training and deployment
  * Face tracking techniques
  * OpenCV dashboard development
  * Building user-focused AI applications

______________________________________________________________________________________________

AUTHOR:

Sri Haran

Emotion Mirror pro - Bipolar Factory Assesment
