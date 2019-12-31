# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 17:54:12 2019

@author: AI & ML
"""

import os
import pickle
import numpy as np

files = [x for x in os.listdir(os.getcwd()) if x.endswith('.p')]

#files[0].split('.')[0]

x_train = []
y_train = []
for i in files:
    
    with open(i,'rb') as f:
        data = pickle.load(f)
        print(i,'size',data.shape)
    for j in range(data.shape[0]):
        if j<100:
            x_train.append(data[j])
            y_train.append(i.split('.')[0])

x_train = np.array(x_train)
y_train = np.array(y_train)

with open('images.p','wb') as f:
    pickle.dump(x_train,f)
with open('labels.p','wb') as f:
    pickle.dump(y_train,f)