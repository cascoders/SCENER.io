#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import cv2
import matplotlib.pyplot as plt

model = tf.keras.models.load_model('high_accuracy_complex.h5')
file_src = '../intel-image-classification/speed/'
file_name_li = os.listdir(file_src)
ans_list = {}
mapping_classifier={0:"building",1:"forest",2:"glacier",3:"mountain",4:"sea",5:"street"}
for i in file_name_li[:16]:
    print(i)
    img = cv2.imread(os.path.join(file_src,i))
    img = cv2.resize(img,(150,150))
    img = np.expand_dims(img,axis = 0)
    ans = model.predict(img)
    ans = np.argmax(ans)
    ans_list[i] = mapping_classifier[ans]

print(ans_list)
