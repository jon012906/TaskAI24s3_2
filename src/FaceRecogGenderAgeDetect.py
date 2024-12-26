# Suppress macOS warning
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

import cv2
import numpy as np
import json
import os
import logging
from config import CAMERA, FACE_DETECTION, PATHS, CONFIDENCE_THRESHOLD

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize the camera with error handling
def initialize_camera(camera_index: int = 0) -> cv2.VideoCapture:
    try:
        cam = cv2.VideoCapture(camera_index)
        if not cam.isOpened():
            logger.error("Could not open webcam")
            return None
            
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA['width'])
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA['height'])
        return cam
    except Exception as e:
        logger.error(f"Error initializing camera: {e}")
        return None

# Load name mappings from JSON file
def load_names(filename: str) -> dict:
    try:
        names_json = {}
        if os.path.exists(filename):
            with open(filename, 'r') as fs:
                content = fs.read().strip()
                if content:
                    names_json = json.loads(content)
        return names_json
    except Exception as e:
        logger.error(f"Error loading names: {e}")
        return {}

# Load pre-trained models for age and gender detection
age_model = "model/age_net.caffemodel"
age_proto = "model/age_deploy.prototxt"
gender_model = "model/gender_net.caffemodel"
gender_proto = "model/gender_deploy.prototxt"

# Load models using OpenCV DNN
age_net = cv2.dnn.readNetFromCaffe(age_proto, age_model)
gender_net = cv2.dnn.readNetFromCaffe(gender_proto, gender_model)

# Define model inputs
AGE_LIST = ['(0-2)','(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_LIST = ['Male', 'Female']

if __name__ == "__main__":
    try:
        logger.info("Starting face recognition system...")
        
        # Initialize face recognizer
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        if not os.path.exists(PATHS['trainer_file']):
            raise ValueError("Trainer file not found. Please train the model first.")
        recognizer.read(PATHS['trainer_file'])
        
        # Load face cascade classifier
        face_cascade = cv2.CascadeClassifier(PATHS['cascade_file'])
        if face_cascade.empty():
            raise ValueError("Error loading cascade classifier")
        
        # Initialize camera
        cam = initialize_camera(CAMERA['index'])
        if cam is None:
            raise ValueError("Failed to initialize camera")
        
        # Load names
        names = load_names(PATHS['names_file'])
        if not names:
            logger.warning("No names loaded, recognition will be limited")
        
        logger.info("Face recognition started. Press 'ESC' to exit.")
        
        while True:
            ret, img = cam.read()
            if not ret:
                logger.warning("Failed to grab frame")
                continue
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=FACE_DETECTION['scale_factor'],
                minNeighbors=FACE_DETECTION['min_neighbors'],
                minSize=FACE_DETECTION['min_size']
            )
            
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Recognize the face
                id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
                
                # Check confidence and display result
                if confidence >= CONFIDENCE_THRESHOLD:
                    name = names.get(str(id), "Unknown")
                    confidence_text = f"{confidence:.1f}%"
                else:
                    name = "Unknown"
                    confidence_text = "N/A"
                
                face_roi = img[y:y+h, x:x+w]
                blob = cv2.dnn.blobFromImage(face_roi, 1.0, (227, 227), 
                                            (78.4263377603, 87.7689143744, 114.895847746), 
                                            swapRB=False)

                # Gender prediction
                gender_net.setInput(blob)
                gender_preds = gender_net.forward()
                gender = GENDER_LIST[gender_preds[0].argmax()]

                # Age prediction
                age_net.setInput(blob)
                age_preds = age_net.forward()
                age = AGE_LIST[age_preds[0].argmax()]
                
                # Adjust positions
                name_position = (x + 5, y - 10) 
                age_position = (x + 5, y + h + 20)  
                gender_position = (x + 5, y + h + 45) 
                confidence_position = (x + 5, y + h - 10)  

                # Display Name 
                cv2.putText(img, name, name_position, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                # Display Age 
                cv2.putText(img, age, age_position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (204, 0, 0), 1)
                # Display Gender 
                cv2.putText(img, gender, gender_position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (204, 0, 0), 1)
                # Display Confidence
                cv2.putText(img, confidence_text, confidence_position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            
            cv2.imshow('Face Recognition', img)
            
            # Check for ESC key
            if cv2.waitKey(1) & 0xFF == 27:
                break
        
        logger.info("Face recognition stopped")
        
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        
    finally:
        if 'cam' in locals():
            cam.release()
        cv2.destroyAllWindows()
