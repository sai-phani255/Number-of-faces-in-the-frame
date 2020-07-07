import cv2
import numpy as np

face = cv2.CascadeClassifier('haar cascade files\haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL

while(True):
    ret, frame = cap.read()
    height,width = frame.shape[:2] 

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.05)

    cv2.rectangle(frame, (0,height-50) , (200,height) , (0,0,0) , thickness=cv2.FILLED )

    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y) , (x+w,y+h) , (255,0,0) , 4 )
        
        cv2.putText(frame,"Stud ="+str(len(faces)),(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)

    cv2.imshow('frame',frame)
    key=cv2.waitKey(1)
    
    if key == ord('s'):
        break
cap.release()
cv2.destroyAllWindows()

            
