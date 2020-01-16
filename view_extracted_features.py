# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 10:03:34 2020

@author: AI & ML
"""

from keras.models import load_model, Model
from keras.layers import Input
import cv2
import matplotlib.pyplot as plt

model = load_model('face_model_multi.h5')

print(model.summary())

img = cv2.imread(r"C:\Users\Dell\Desktop\Datasets\standard_test_images\lena_color_256.tif")

def preprocessing(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img,(200,200))
    img = cv2.equalizeHist(img)
    img = img.reshape(1,200,200,1)
    img = img/255
    return img

img = preprocessing(img)

model2 = Model(model.inputs,model.layers[0].output)
features = model2.predict(img)

plt.figure(figsize=(10,50))
for i in range(16):
    plt.subplot(4,4,i+1)
    plt.imshow(features[0,:,:,i])
    plt.axis('off')
    plt.show()