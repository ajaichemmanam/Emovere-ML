import cv2

# starting video streaming
cv2.namedWindow('Emovere')
video_capture = cv2.VideoCapture(0)
while(True):
    bgr_image = video_capture.read()[1]
    cv2.imshow("Emovere", bgr_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
