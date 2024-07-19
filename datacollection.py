import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap=cv2.VideoCapture(0)
detector=HandDetector(maxHands=1)
offset=20
imgSize=300
counter=0

folder="/Users/Astha/OneDrive/Desktop/Sign Language Detection/Data/I like you"

while True :
    success,img=cap.read()
    hands,img=detector.findHands(img)
    if hands:
        hand=hands [0]
        x,y,w,h=hand['bbox']

        imgwhite=np.ones((imgSize,imgSize,3),np.uint8)*255

        imgcrop=img[y-offset:y+h+offset,x-offset :x+w+offset]
        imgcropShape=imgcrop.shape

        aspectratio=h/w

        if aspectratio>1:
            k=imgSize/h
            wCal=math.ceil(k*w)
            imgResize=cv2.resize(imgcrop,(wCal,imgSize))
            imgResizeShape=imgResize.shape
            wGap=math.ceil((imgSize-wCal )/2)
            imgwhite[: ,wGap:wCal +wGap]=imgResize

        else:
            k=imgSize/w    
            hCal=math.ceil(k*h)
            imgResize=cv2.resize(imgcrop,(imgSize,hCal))
            imgResizeShape=imgResize.shape
            hGap=math.ceil((imgSize-hCal )/2)
            imgwhite[hGap:hCal +hGap, :]=imgResize

        cv2.imshow('ImageCrop',imgcrop)
        cv2.imshow('ImageWhite',imgwhite)

    cv2.imshow("Image",img)
    key=cv2.waitKey(2)
    if key==ord("s"):
        counter+=1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg',imgwhite)
        print(counter)


