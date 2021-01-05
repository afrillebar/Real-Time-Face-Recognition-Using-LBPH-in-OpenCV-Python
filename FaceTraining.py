import cv2
import numpy as np
from PIL import Image
import os

recognizer = cv2.face.LBPHFaceRecognizer_create()

# Path for face image database
path="dataset"

def getImagesWithID(path):
    # get the path of all the files in the folder
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    faces=[]
    # creating empty ID list
    IDs=[]
    for imagePath in imagePaths:
        # loading the image and converting it to grayscale
        faceImg=Image.open(imagePath).convert('L')
        # converting into numpy array
        faceNp=np.array(faceImg,'uint8')
        # getting the Id from the image
        ID=int(os.path.split(imagePath)[-1].split('.')[1])
        faces.append(faceNp)
        IDs.append(ID)
        cv2.imshow("training",faceNp)
        cv2.waitKey(10)
    return np.array(IDs), faces

Ids, faces = getImagesWithID(path)
recognizer.train(faces, Ids)
recognizer.save('trainer/trainer.yml')