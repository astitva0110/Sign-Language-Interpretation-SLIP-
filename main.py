import cv2
import numpy as np
import math
#import pyttsx3
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier

# Initialize Text-to-Speech engine
#engine = pyttsx3.init()

# Initialize video capture
cap = cv2.VideoCapture(0)

# Check if the camera is opened successfully
if not cap.isOpened():
    print("Error: Unable to open the camera.")
    exit()

# Initialize hand detector and classifier
detector = HandDetector(maxHands=1)
classifier = Classifier("Models/keras_model.h5", "Models/labels.txt")

offset = 20
imgSize = 300
prev_label_index = -1

labels = ["Hello", "How are you", "I like you", "No", "Okay","Please", "Thank you","Victory","Washroom","Yes"]

# Set up parameters for smoother resizing
interpolation = cv2.INTER_AREA  # Use INTER_AREA for shrinking (resizing)
border_color = (255, 255, 255)  # White border color for padding

while True:
    try:
        success, img = cap.read()
        if not success:
            print("Error: Failed to capture frame from camera.")
            break

        imgOutput = img.copy()
        hands, _ = detector.findHands(img)  # Use '_' to discard unnecessary output
        speak_label = False

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

            # Resize the cropped image to imgSize with padding if necessary
            if h > w:
                target_height = imgSize
                target_width = math.ceil(w * (imgSize / h))
                pad_width = (imgSize - target_width) // 2
                imgResize = cv2.resize(imgCrop, (target_width, target_height), interpolation=interpolation)
                imgResize = cv2.copyMakeBorder(imgResize, 0, 0, pad_width, pad_width, cv2.BORDER_CONSTANT,
                                                value=border_color)
            else:
                target_width = imgSize
                target_height = math.ceil(h * (imgSize / w))
                pad_height = (imgSize - target_height) // 2
                imgResize = cv2.resize(imgCrop, (target_width, target_height), interpolation=interpolation)
                imgResize = cv2.copyMakeBorder(imgResize, pad_height, pad_height, 0, 0, cv2.BORDER_CONSTANT,
                                                value=border_color)

            prediction, index = classifier.getPrediction(imgResize, draw=False)

            cv2.rectangle(imgOutput, (x - offset, y - offset - 70), (x - offset + 400, y - offset + 60 - 50),
                          (0, 255, 0), cv2.FILLED)

            cv2.putText(imgOutput, labels[index], (x - offset, y - offset), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0),
                        2)
            cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (0, 255, 0), 4)

            if prev_label_index != index:
                speak_label = True
                prev_label_index = index

            #if speak_label:
                # Speak the recognized label only when it changes
             #   engine.say(labels[index])
              #  engine.runAndWait()

        cv2.imshow('Image', imgOutput)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    except Exception as e:
        print("Error:", e)
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
