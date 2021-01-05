import cv2, os, datetime, time
import numpy as np
import pandas as pd

recognizer = cv2.face.LBPHFaceRecognizer_create()
# Reading the trained model
recognizer.read('trainer/trainer.yml')
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
# getting the name from "User.csv"
df=pd.read_csv("User\\User.csv")

# font
font = cv2.FONT_HERSHEY_SIMPLEX

#iniciate id counter
id = 0

# Initialize and start realtime video capture
cap = cv2.VideoCapture(0)

while True:
    # reads frames from a camera
    ret, img = cap.read()
    img = cv2.flip(img, 1)

    # convert to gray scale of each frames
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Detects faces of different sizes in the input image 
    faces = face_cascade.detectMultiScale(gray, 1.2, 5) 

    for(x,y,w,h) in faces:
        # To draw a rectangle in a face 
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)

        Id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

        if (confidence < 100):
            aa=df.loc[df['face_id'] == Id]['face_name'].values
            tt=str(Id)+"-"+aa
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            Id = "unknown"
            tt=str(Id)
            confidence = "  {0}%".format(round(100 - confidence))
        
        cv2.putText(img, str(tt), (x+5,y-5), font, 1, (255,255,0), 1)
        cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)  
        
    cv2.imshow('img',img) 

    k = cv2.waitKey(10) & 0xff # Press ESC for exiting video
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
