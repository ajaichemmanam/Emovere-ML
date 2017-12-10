import cv2

# starting video streaming
cv2.namedWindow('Emovere')
video_capture = cv2.VideoCapture(0)
while(True):
    bgr_image = video_capture.read()[1]
    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    cv2.imshow("Emovere", bgr_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
