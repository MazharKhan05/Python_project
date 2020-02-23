import cv2
face_cascade = cv2.CascadeClassifier(r"/home/aiktc/Desktop/PythonWorkshop/Classifiers/haarcascade_frontalface_default.xml")
body_cascade = cv2.CascadeClassifier(r"/home/aiktc/Desktop/PythonWorkshop/Classifiers/haarcascade_frontalface_default.xml")
video = cv2.VideoCapture(r"/home/aiktc/Desktop/PythonWorkshop/faceDetection.mp4")

print(type(video))

check=True
while(check!=False):
    check, frame = video.read()
    grey_img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(grey_img,scaleFactor = 1.25,minNeighbors =14)
    body = body_cascade.detectMultiScale(grey_img,scaleFactor = 1.25,minNeighbors = 14)
    for x,y,w,h in faces:
        frame = cv2.rectangle(frame,(x,y),(x+w,y+w),(0,255,0),3)
    for x,y,w,h in body:
        frame = cv2.rectangle(frame,(x,y),(x+w,y+w),(0,255,0),3)
    cv2.imshow("Video",frame)
    key=cv2.waitKey(1)#returns the key pressed by the user
    if(key == ord('q')):
        break

cv2.destroyAllWindows()
video.release()