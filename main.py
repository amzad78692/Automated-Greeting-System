import cv2
import face_recognition as fr
import numpy as np
import os
import pyttsx3

def findEncodes(images):
    encodeList = []

    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = fr.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def Speak(text):
    text = str(text)
    text = 'Hello ' + text + '. Welcome to Subharti Institute of Technology and Engineering'
    engine = pyttsx3.init('sapi5')
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[3].id)
    engine.setProperty('rate', 170)
    engine.say(text)
    engine.runAndWait()

imgList = []
names = []

list = os.listdir('Images')

for cl in list:
    curimg = cv2.imread('Images/'+cl)
    imgList.append(curimg)
    names.append(os.path.splitext(cl)[0])

knownList = findEncodes(imgList)

cap = cv2.VideoCapture(2)

while True:
    success, img = cap.read()

    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    faces = fr.face_locations(imgS)
    encodes = fr.face_encodings(imgS, faces)

    for encodeface, faceLoc in zip(encodes, faces):
        matches = fr.compare_faces(knownList, encodeface)
        faceDis = fr.face_distance(knownList, encodeface)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = names[matchIndex].upper()
            y1, x2, y2, x1 = faceLoc
            # y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            Speak(name)


    cv2.imshow("Camera", img)
    cv2.waitKey(1)