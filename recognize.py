# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 17:30:16 2019

@author: AI & ML
"""
import urllib
import cv2
import numpy as np
from keras.models import load_model

classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = load_model('face_model_multi.h5')
URL = "http://25.38.153.116:8080/shot.jpg"


def get_pred_label(pred):
    labels = ['Ashutosh',
              'Shamaun',
              'Taras',
              'uma']
    return labels[pred]

def preprocess(img):
    img = cv2.resize(img,(200,200))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img.reshape(1,200,200,1)
    img = img/255
    return img
    

ret = True
while ret:
    img = urllib.request.urlopen(URL)
    image = np.array(bytearray(img.read()),dtype=np.uint8)
    frame = cv2.imdecode(image,-1)

    faces = classifier.detectMultiScale(frame,1.3,5)
    
    for x,y,w,h in faces:
        face = frame[y:y+h+10,x:x+h+10]
        cv2.rectangle(frame,(x,y),
                      (x+w,y+h),(0,255,0),5)
        cv2.putText(frame,get_pred_label(model.predict_classes(preprocess(face))[0]),(x,y),
                                         cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),2)
    cv2.imshow('video',frame)
        
    if cv2.waitKey(1)==ord('q'):
        break
cv2.destroyAllWindows()



