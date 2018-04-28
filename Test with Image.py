from statistics import mode

import cv2
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

def preprocess_input(x, v2=True):
    x = x.astype('float32')
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x

def get_labels():
        return {0:'angry',1:'disgust',2:'fear',3:'happy',
                4:'sad',5:'surprise',6:'neutral'}

def load_detection_model(model_path):
    detection_model = cv2.CascadeClassifier(model_path)
    return detection_model

def detect_faces(detection_model, gray_image_array):
    return detection_model.detectMultiScale(gray_image_array, 1.3, 5)

def apply_offsets(face_coordinates, offsets):
    x, y, width, height = face_coordinates
    x_off, y_off = offsets
    return (x - x_off, x + width + x_off, y - y_off, y + height + y_off)
   
def emotion_analysis(emotions):
    objects = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
    y_pos = np.arange(len(objects))
    plt.bar(y_pos, emotions, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('Percentage')
    plt.title('Emotion')
    plt.show()    

# parameters for loading data and images
detection_model_path = 'haar/haarcascade_frontalface_default.xml'
emotion_model_path = 'training_output/fer2013_mini_XCEPTION.94-0.66.hdf5'
emotion_labels = get_labels()

# hyper-parameters for bounding boxes shape
frame_window = 10
emotion_offsets = (20, 40)

# loading models
face_detection = load_detection_model(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)

# getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]

# starting lists for calculating modes
#the mode of the list is the item that occurs most often.
emotion_window = []
img =cv2.imread('TestImage.jpg')
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = detect_faces(face_detection, gray_image)
cv2.imshow('image',img)
for face_coordinates in faces:
    x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
    gray_face = gray_image[y1:y2, x1:x2]
    try:
        gray_face = cv2.resize(gray_face, (emotion_target_size))
        #cv2.imshow('Haar_Output',gray_face)
    except:
        continue
gray_face = preprocess_input(gray_face, True)
gray_face = np.expand_dims(gray_face, 0)
gray_face = np.expand_dims(gray_face, -1)
custom = emotion_classifier.predict(gray_face)
emotion_analysis(custom[0])
