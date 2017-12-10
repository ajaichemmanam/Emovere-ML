import cv2

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

# load models
face_detection = load_haar(haar_cascade_path)

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
        cv2.imshow("HaarOutput", gray_face)
    cv2.imshow("Emovere", bgr_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
