import cv2
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
import datetime
import os



# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Map emotions to emoji image paths (update these paths as needed)
EMOJI_PATHS = {
    'happy': 'Emotions-Recognition-master/images/happy.png',
    'sad': 'Emotions-Recognition-master/images/sad.png',
    'angry': 'Emotions-Recognition-master/images/angry.png',
    'neutral': 'Emotions-Recognition-master/images/neutral.png',
    'surprise': 'Emotions-Recognition-master/images/surprise.png'
}

# Create or clear log file
log_file = "emotion_log.txt"
if not os.path.exists(log_file):
    open(log_file, "w").close()

def log_emotion(emotion):
    with open(log_file, "a") as f:
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        f.write(f"{timestamp}: {emotion}\n")

# Load emotion recognition model (simplified - you'll need to adapt this to your Emotions-Recognition project)
def load_emotion_model():
    # This is a placeholder - replace with actual model loading from your Emotions-Recognition project
    print("Loading emotion recognition model...")
    # Example: return cv2.face.LBPHFaceRecognizer_create()
    return None

def predict_emotion(face_img, model):
    # Placeholder - replace with your actual emotion prediction code
    # This should return one of the emotions in EMOJI_PATHS.keys()
    return "neutral"  # Default

# Tkinter Setup
root = tk.Tk()
root.title("Face Emotion Detector with Emoji")
root.geometry("900x600")

video_label = tk.Label(root)
video_label.pack()

emoji_label = tk.Label(root)
emoji_label.pack()

cap = cv2.VideoCapture(0)
emotion_model = load_emotion_model()

def show_frame():
    ret, frame = cap.read()
    if not ret:
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    for (x, y, w, h) in faces:
        # Draw rectangle around face
        cv2.rectangle(rgb, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Extract face ROI
        face_roi = gray[y:y+h, x:x+w]
        
        # Predict emotion (replace with your actual prediction code)
        emotion = predict_emotion(face_roi, emotion_model)
        
        # Display emotion text
        cv2.putText(rgb, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
        
        # Display emoji
        try:
            emoji_img = Image.open(EMOJI_PATHS.get(emotion, EMOJI_PATHS['neutral']))
            emoji_img = emoji_img.resize((50, 50))
            emoji_photo = ImageTk.PhotoImage(emoji_img)
            emoji_label.config(image=emoji_photo)
            emoji_label.image = emoji_photo
        except Exception as e:
            print(f"Error loading emoji: {e}")
        
        log_emotion(emotion)

    # Convert to ImageTk format
    img = Image.fromarray(rgb)
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    root.after(30, show_frame)

show_frame()
root.mainloop()
cap.release()
