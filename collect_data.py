# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 12:25:13 2019

@author: AI & ML
"""
import urllib
import cv2
import numpy as np

classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
URL = "http://56.108.111.245:8080/shot.jpg"

def preprocess(img):
    img = cv2.resize(img,(200,200))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img
data = []
ret = True
while ret:
    img = urllib.request.urlopen(URL)
    image = np.array(bytearray(img.read()),dtype=np.uint8)
    frame = cv2.imdecode(image,-1)

    faces = classifier.detectMultiScale(frame,1.3,5)

    if faces is not None:    
        for x,y,w,h in faces:
            face = frame[y:y+h+50,x:x+h+50]
            cv2.imshow('face',preprocess(face))
            if len(data)<=100:
                data.append(preprocess(face))
            else:
                cv2.putText(frame,'done',(100,100),
                            cv2.FONT_HERSHEY_PLAIN,4,
                            (255,255,255),5)
        
    cv2.imshow('video',frame)

    if cv2.waitKey(1)==ord('q'):
        break
cv2.destroyAllWindows()

data = np.array(data)

name = input('enter the name of person: ')
import pickle
with open(name+'.p','wb') as f:
    pickle.dump(data,f)






