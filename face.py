import cv2
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
import datetime
import os
import time

# --- Configuration ---
FACE_CASCADE_PATH = 'haarcascade_frontalface_default.xml'
EMOTION_MODEL_PATH = 'emotion-ferplus-8.onnx'
EMOJI_DIR = 'images'
LOG_FILE = "emotion_log.txt"
LOG_INTERVAL = 2  # Log every 2 seconds to avoid bloat

EMOJI_PATHS = {
    'happy': os.path.join(EMOJI_DIR, 'happy.png'),
    'sad': os.path.join(EMOJI_DIR, 'sad.png'),
    'angry': os.path.join(EMOJI_DIR, 'angry.png'),
    'neutral': os.path.join(EMOJI_DIR, 'neutral.png'),
    'surprise': os.path.join(EMOJI_DIR, 'surprise.png')
}

# FERPlus labels: ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt']
EMOTION_LABELS = ['neutral', 'happy', 'surprise', 'sad', 'angry', 'sad', 'sad', 'neutral']

# --- State ---
last_log_time = 0
preloaded_emojis = {}

# --- Initialization ---
# Load Haar Cascade for face detection
if not os.path.exists(FACE_CASCADE_PATH):
    print(f"Error: {FACE_CASCADE_PATH} not found!")
    if os.path.exists('facedetecter.xml'):
        FACE_CASCADE_PATH = 'facedetecter.xml'

face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)

def preload_images():
    """Pre-load and resize emoji images for performance."""
    print("Pre-loading emojis...")
    for emotion, path in EMOJI_PATHS.items():
        try:
            if os.path.exists(path):
                img = Image.open(path).convert("RGBA")
                img = img.resize((100, 100), Image.Resampling.LANCZOS)
                preloaded_emojis[emotion] = ImageTk.PhotoImage(img)
            else:
                print(f"Warning: Emoji path {path} does not exist.")
        except Exception as e:
            print(f"Error pre-loading {emotion}: {e}")

def log_emotion(emotion):
    """Log emotion to file with throttling."""
    global last_log_time
    current_time = time.time()
    
    if current_time - last_log_time >= LOG_INTERVAL:
        with open(LOG_FILE, "a") as f:
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"{timestamp}: {emotion}\n")
        last_log_time = current_time

def load_emotion_model():
    """Load the ONNX emotion recognition model."""
    if not os.path.exists(EMOTION_MODEL_PATH):
        print(f"Error: {EMOTION_MODEL_PATH} not found. Emotion detection will be disabled.")
        return None
    try:
        print(f"Loading emotion model from {EMOTION_MODEL_PATH}...")
        net = cv2.dnn.readNetFromONNX(EMOTION_MODEL_PATH)
        return net
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def predict_emotion(face_img, net):
    """
    Predict emotion using the ONNX model.
    FERPlus model expects 64x64 grayscale input.
    """
    if net is None:
        return "neutral"
    
    try:
        # Preprocess: Resize to 64x64 and normalize
        blob = cv2.dnn.blobFromImage(face_img, 1.0, (64, 64), (0), swapRB=False, crop=False)
        net.setInput(blob)
        preds = net.forward()
        
        # Get index of highest probability
        idx = np.argmax(preds)
        if idx < len(EMOTION_LABELS):
            return EMOTION_LABELS[idx]
    except Exception as e:
        print(f"Inference error: {e}")
        
    return "neutral"

# --- Tkinter Setup ---
root = tk.Tk()
root.title("Face Emotion Detector")
root.geometry("1000x800")
root.configure(bg="#2c3e50")

# Header
header = tk.Label(root, text="Real-Time Face Emotion Detector", font=("Helvetica", 24, "bold"), fg="white", bg="#2c3e50", pady=10)
header.pack()

# Main Container
main_frame = tk.Frame(root, bg="#2c3e50")
main_frame.pack(expand=True, fill="both", padx=20, pady=10)

# Video Feed
video_label = tk.Label(main_frame, bg="black", borderwidth=4, relief="ridge")
video_label.grid(row=0, column=0, padx=10)

# Sidebar for Emoji
sidebar = tk.Frame(main_frame, bg="#34495e", width=300, height=500, borderwidth=2, relief="sunken")
sidebar.grid(row=0, column=1, padx=10, sticky="nsew")
sidebar.grid_propagate(False)

emoji_title = tk.Label(sidebar, text="Detected Emotion", font=("Helvetica", 14), fg="#bdc3c7", bg="#34495e", pady=20)
emoji_title.pack()

emoji_display = tk.Label(sidebar, bg="#34495e")
emoji_display.pack(expand=True)

emotion_text = tk.Label(sidebar, text="WAITING...", font=("Helvetica", 22, "bold"), fg="#2ecc71", bg="#34495e")
emotion_text.pack(pady=30)

# --- Video Capture ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video source.")
    error_label = tk.Label(root, text="Error: Camera not found!", fg="#e74c3c", font=("Helvetica", 16), bg="#2c3e50")
    error_label.pack()

preload_images()
emotion_net = load_emotion_model()

def show_frame():
    ret, frame = cap.read()
    if not ret:
        root.after(10, show_frame)
        return

    # Mirror the frame
    frame = cv2.flip(frame, 1)
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(60, 60)
    )

    current_detect = "neutral"

    for (x, y, w, h) in faces:
        # Draw sleek rectangle
        cv2.rectangle(rgb, (x, y), (x+w, y+h), (46, 204, 113), 3)
        
        # Extract face ROI for emotion
        face_roi = gray[y:y+h, x:x+w]
        
        # Predict emotion
        current_detect = predict_emotion(face_roi, emotion_net)
        
        # Display emotion text on video
        cv2.putText(rgb, current_detect.upper(), (x, y-15), cv2.FONT_HERSHEY_DUPLEX, 1.0, (46, 204, 113), 2)
        
        log_emotion(current_detect)

    # Update UI
    if len(faces) > 0:
        emotion_text.config(text=current_detect.upper())
        if current_detect in preloaded_emojis:
            emoji_display.config(image=preloaded_emojis[current_detect])
    else:
        emotion_text.config(text="WAITING...")
        if "neutral" in preloaded_emojis:
             emoji_display.config(image=preloaded_emojis["neutral"])

    # Update video label
    img = Image.fromarray(rgb)
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    root.after(15, show_frame)

def on_closing():
    cap.release()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)
show_frame()
root.mainloop()


