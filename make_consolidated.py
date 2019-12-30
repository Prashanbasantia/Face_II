# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 22:45:13 2019

@author: LENOVO
"""

import pickle
import numpy as np
import os


images = []
labels = []
for i in [x for x in os.listdir(os.getcwd()) if x.endswith('.p')]:
    with open(i,'rb') as f:
        arr = pickle.load(f)
        for j in arr:
            images.append(j)
            labels.append(i.split('.')[0])

images = np.array(images)
labels = np.array(labels)
with open('images.p','wb') as f:
    pickle.dump(images,f)
with open('labels.p','wb') as f:
    pickle.dump(labels,f)

