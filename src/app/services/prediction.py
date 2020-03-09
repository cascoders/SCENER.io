#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import cv2
import matplotlib.pyplot as plt
import boto3

def predict_res():
    ACCESS_KEY = "AKIASKDKDS3L3CMBH6F6"
    SECRET_KEY = "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"

    s3 = boto3.client(
        's3',
        aws_access_key_id = ACCESS_KEY,
        aws_secret_access_key = SECRET_KEY)


    bucket_path = "new-codeshastra-test-bucket"
    temp_folder = "test_folder"

    li = s3.list_objects(
        Bucket=bucket_path
        )

    total_folders = []
    for i in range(len(li['Contents'])):
        total_folders.append(li['Contents'][i]['Key'])
        if('test_folder' not in li['Contents'][i]['Key']):
            total_folders.remove(li['Contents'][i]['Key'])

    total_folders.pop(0)

    li_model = []
    for i in range(len(total_folders)):
        li_model.append(total_folders[i][12:])

    li_model_path = total_folders

    for i in range(len(li_model_path)):
        s3.download_file(
            Bucket = bucket_path,
            Key = li_model_path[i],
            Filename = li_model[i])



    '''----------IMAGE READ CODE FINISH----------'''


    model = tf.keras.models.load_model('/home/ubuntu/df/intel_image_classification/src/app/services/res_model.h5')
    mapping_classifier={0:"building",1:"forest",2:"glacier",3:"mountain",4:"sea",5:"street"}
    ans_list = {}
    file_src = os.getcwd();
    for i in os.listdir(file_src):
        if('jpg' in i or 'png' in i):
            img = cv2.imread(os.path.join(file_src,i))
            img = cv2.resize(img,(150,150))
            img = np.expand_dims(img,axis = 0)
            ans = model.predict(img)
            ans = np.argmax(ans)
            ans_list[i] = mapping_classifier[ans]


    for i in os.listdir(file_src):
        if('jpg' in i or 'png' in i):
            os.remove(i)

    return ans_list
