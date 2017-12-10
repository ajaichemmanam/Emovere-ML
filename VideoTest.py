import cv2
from tensorflow.python.keras.models import load_model
import numpy as np

def preprocess_input(x): #to convert pixel values to range -1 to +1
    x = x.astype('float32')
    x = x / 255.0
    x = x - 0.5
    x = x * 2.0
    return x

def load_haar(path):
    haar_model = cv2.CascadeClassifier(path)
    return haar_model

def detect_faces(detection_model, gray_image):
    return detection_model.detectMultiScale(gray_image, 1.3, 5)

def apply_offsets(face_coordinates, offsets):
    x, y, width, height = face_coordinates
    x_off, y_off = offsets
    return (x - x_off, x + width + x_off, y - y_off, y + height + y_off)

face_offsets = (20, 40)

# Path to model
haar_cascade_path = 'haar/haarcascade_frontalface_default.xml'
emotion_model_path = 'training_output/fer2013_mini_XCEPTION.94-0.66.hdf5'

# load models
face_detection = load_haar(haar_cascade_path)
emotion_classifier = load_model(emotion_model_path, compile=False)

# getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]

# starting video streaming
cv2.namedWindow('Emovere')
video_capture = cv2.VideoCapture(0)
while(True):
    bgr_image = video_capture.read()[1]
    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    faces = detect_faces(face_detection, gray_image)
    #print(faces)   #Print Face-co-ordiates recieved from haar cascade
    for face_coordinates in faces:
        x1, x2, y1, y2 = apply_offsets(face_coordinates, face_offsets)
        gray_face = gray_image[y1:y2, x1:x2]
        cv2.imshow("HaarOutput", gray_face) #Show Output of Haar
        try:
            gray_face = cv2.resize(gray_face, (emotion_target_size))
            cv2.imshow('Haar_Resized',gray_face)   #Show Resized Image
        except:
            continue
        gray_face = preprocess_input(gray_face)
        cv2.imshow('Preprocessed_face',gray_face)    #Show Image after preprocessing
        gray_face = np.expand_dims(gray_face, 0) # Image has only 2 Dimension, But we require 4 dimension input to our model
        gray_face = np.expand_dims(gray_face, -1)
        
        emotion_prediction = emotion_classifier.predict(gray_face)
        print(emotion_prediction)  #Show Model Prediction
        emotion_probability = np.max(emotion_prediction)
        print(emotion_probability) # Show Maximum Probability
        emotion_label_arg = np.argmax(emotion_prediction)   
        print(emotion_label_arg)
    cv2.imshow("Emovere", bgr_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
