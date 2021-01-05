import cv2, csv, os

# Trained XML file for detecting faces
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# capture frames from a camera
cap = cv2.VideoCapture(0)

# Both ID and Name is used for recognising the Image 
face_id = input('\n face id :  ')
face_name = input('\n face name :  ')

# Initialize individual sampling face count
count = 0

while True:
    # reads frames from a camera
    ret, img = cap.read(0)

    # convert to gray scale of each frames
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detects faces of different sizes in the input image 
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        # To draw a rectangle in a face 
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        
        # Save the captured image into the datasets folder
        if count %2 ==0: # every 5 seconds
            cv2.imwrite("dataset/"+ str(face_name) +'.'+ str(face_id) + '.' + str(count) + ".jpg", gray[y:y + h, x:x + w])
        count += 1

        cv2.imshow('image', img)

    k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break
    elif count >= 60: # Take 30 face sample and stop video
         break

cap.release()
cv2.destroyAllWindows()
row = [face_id , face_name]
with open('User\\User.csv','a+') as csvFile:
    writer = csv.writer(csvFile)
    # Entry of the row in csv file 
    writer.writerow(row)
csvFile.close()