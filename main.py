# main.py

# Imports
import cv2
import numpy as np
# Import TensorFlow and Keras
import tensorflow as tf
from tensorflow.keras.models import model_from_json
import os # For checking file paths

# --- Constants ---
# List of emotion labels corresponding to the output classes of the Keras model.
# Ensure this order matches the training output of the model defined in .json/.h5 files.
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
# Input image size expected by the emotion recognition model (adjust if your Keras model expects different).
IMG_SIZE = 48
# Path to the Haar Cascade XML file for face detection.
# --- IMPORTANT USER INSTRUCTION ---
# Ensure 'haarcascade_frontalface_default.xml' is in the same directory or update path.
HAAR_CASCADE_PATH = 'haarcascade_frontalface_default.xml'
# Paths for the Keras model architecture and weights.
# --- IMPORTANT USER INSTRUCTION ---
# Ensure 'emotion_model.json' (architecture) and 'emotion_model.h5' (weights)
# are in the same directory as this script.
MODEL_JSON_PATH = 'emotion_model.json'
MODEL_H5_PATH = 'emotion_model.h5'

# --- (Device Setup Removed - TensorFlow/Keras handles device placement) ---
# print(f"Using device: {device}") # Removed PyTorch device logic

# --- (PyTorch Emotion Model Definition Removed) ---
# class EmotionCNN(nn.Module): ... (Removed)

# --- (PyTorch Preprocessing Definition Removed) ---
# emotion_preprocess = transforms.Compose([...]) # Removed

# --- Load Resources ---
def load_resources():
    """Loads the Haar Cascade classifier and the Keras Emotion model."""
    # Load Haar Cascade for face detection (same as before)
    face_cascade = None
    if not os.path.exists(HAAR_CASCADE_PATH):
        print(f"--- ERROR ---")
        print(f"Haar Cascade file not found at: {HAAR_CASCADE_PATH}")
        print(f"Please ensure '{os.path.basename(HAAR_CASCADE_PATH)}' is in the same directory.")
        print(f"Face detection will not work.")
        print(f"-------------")
    else:
        face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
        if face_cascade.empty():
             print(f"--- ERROR ---")
             print(f"Failed to load Haar Cascade from: {HAAR_CASCADE_PATH}")
             print(f"The file might be corrupted or incompatible.")
             print(f"Face detection will not work.")
             print(f"-------------")
             face_cascade = None
        else:
             print("Haar Cascade loaded successfully.")

    # Load Keras Emotion Recognition Model from JSON and H5 files
    emotion_model = None
    if not os.path.exists(MODEL_JSON_PATH) or not os.path.exists(MODEL_H5_PATH):
        print(f"--- WARNING ---")
        if not os.path.exists(MODEL_JSON_PATH):
             print(f"Keras model architecture file not found at: {MODEL_JSON_PATH}")
        if not os.path.exists(MODEL_H5_PATH):
             print(f"Keras model weights file not found at: {MODEL_H5_PATH}")
        print(f"To enable emotion recognition, ensure '{os.path.basename(MODEL_JSON_PATH)}' and ")
        print(f"'{os.path.basename(MODEL_H5_PATH)}' are in the script's directory.")
        print(f"Emotion recognition will be DISABLED.")
        print(f"--------------")
    else:
        try:
            # Load model architecture from JSON file
            with open(MODEL_JSON_PATH, 'r') as json_file:
                loaded_model_json = json_file.read()
            emotion_model = model_from_json(loaded_model_json)

            # Load weights into the model from H5 file
            emotion_model.load_weights(MODEL_H5_PATH)
            print(f"Keras emotion model loaded successfully from {MODEL_JSON_PATH} and {MODEL_H5_PATH}.")

            # Optional: Compile the model if needed for certain operations,
            # but usually not required just for prediction.
            # emotion_model.compile(optimizer='adam', loss='categorical_crossentropy')

        except Exception as e:
            print(f"--- ERROR ---")
            print(f"Failed to load Keras emotion model:")
            print(e)
            print(f"Ensure the JSON/H5 files are valid and compatible.")
            print(f"Emotion recognition will be DISABLED.")
            print(f"-------------")
            emotion_model = None

    return face_cascade, emotion_model

# --- Keras Preprocessing Function (Integrated into main loop for clarity) ---
# No separate function needed, preprocessing steps added directly below.

# --- Main Execution ---
def main():
    """Initializes resources and runs the main detection/recognition loop."""
    print("Loading resources...")
    face_cascade, emotion_model = load_resources()

    # Exit if face detector failed to load
    if face_cascade is None:
        print("Exiting application due to failure in loading Haar Cascade.")
        return

    # Initialize Webcam
    print("Initializing webcam...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("--- ERROR --- Could not open webcam. ---")
        return

    print("Webcam initialized. Starting real-time detection... Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # --- Face Detection ---
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
        )

        # --- Process Each Detected Face ---
        for (x, y, w, h) in faces:
            # Draw bounding box (Blue)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            emotion_label_text = "Emotion N/A"
            label_color = (0, 0, 255) # Red for N/A or errors

            # --- Emotion Recognition (using Keras model if loaded) ---
            if emotion_model is not None:
                try:
                    # Extract Face ROI (Region of Interest) from grayscale frame
                    face_roi_gray = gray_frame[y:y+h, x:x+w]

                    if face_roi_gray.size == 0: continue # Skip if ROI is empty

                    # --- Keras Preprocessing ---
                    # 1. Resize ROI to the target size (e.g., 48x48)
                    resized_roi = cv2.resize(face_roi_gray, (IMG_SIZE, IMG_SIZE))
                    # 2. Convert to float32
                    resized_roi = resized_roi.astype('float32')
                    # 3. Normalize pixel values (common practice: divide by 255.0 for [0,1] range)
                    #    Adjust if your model was trained with different normalization.
                    normalized_roi = resized_roi / 255.0
                    # 4. Reshape for Keras model input (add batch and channel dimensions)
                    #    Expected shape: (batch_size, height, width, channels) -> (1, 48, 48, 1)
                    preprocessed_roi = np.expand_dims(normalized_roi, axis=-1) # Add channel dim
                    preprocessed_roi = np.expand_dims(preprocessed_roi, axis=0)  # Add batch dim


                    # --- Keras Prediction ---
                    # Predict probabilities for each emotion class
                    predictions = emotion_model.predict(preprocessed_roi) # Output is usually [[prob_class1, prob_class2, ...]]

                    # Get the index of the highest probability
                    predicted_idx = np.argmax(predictions[0])
                    confidence_score = np.max(predictions[0])
                    predicted_emotion = EMOTION_LABELS[predicted_idx]

                    emotion_label_text = f"{predicted_emotion} ({confidence_score:.2f})"
                    label_color = (0, 255, 0) # Green for success

                except Exception as e:
                    print(f"Error processing face ROI at ({x},{y},{w},{h}) with Keras model: {e}")
                    emotion_label_text = "Error"
                    # Keep label_color red

            # --- Draw Emotion Label ---
            cv2.putText(frame, emotion_label_text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, label_color, 2)

        # --- Display Output ---
        cv2.imshow('Real-time Face Emotion Recognition (Keras - Press Q to Quit)', frame)

        # --- Exit Condition ---
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("'q' pressed, stopping.")
            break

    # --- Cleanup ---
    print("Releasing webcam and destroying windows...")
    cap.release()
    cv2.destroyAllWindows()
    print("Application finished.")

# --- Script Entry Point ---
if __name__ == '__main__':
    main()