#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, BatchNormalization, LeakyReLU, Flatten, Activation, MaxPool2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau,LearningRateScheduler
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.applications.resnet50 import ResNet50
import tensorflow.keras
from tensorflow.keras.utils import to_categorical
import os
import cv2

def train_res_net(rotation_range_var = 10,
    zoom_range_var = 0.1,
    width_shift_range_var = 0.1,
    height_shift_range_var = 0.1,
    horizontal_flip_var = True,
    vertical_flip_var = True,
    batch_size_var = 32,
    activation_func_var = 'relu',
    epoch_var = 30,
    steps_per_epoch_var = 256,
    validation_steps_var = 256):

    train_dir = '../../dump/intel-image-classification/seg_train/seg_train/'
    test_dir = '../../dump/intel-image-classification/seg_test/seg_test/'


    data_gen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=rotation_range_var,
        zoom_range = zoom_range_var,
        width_shift_range=width_shift_range_var,
        height_shift_range=height_shift_range_var,
        horizontal_flip=horizontal_flip_var,
        vertical_flip=vertical_flip_var)

    train_gen = data_gen.flow_from_directory(
        train_dir,
        target_size = (150,150),
        batch_size=32,
        class_mode = 'categorical'
    )

    test_gen = data_gen.flow_from_directory(
        test_dir,
        target_size=(150,150),
        batch_size=32,
        class_mode='categorical'
    )


    es = EarlyStopping(monitor='val_loss',
                                          mode='min',
                                          verbose=1,
                                          patience=10)

    learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy',
                                                patience=3,
                                                verbose=1,
                                                factor=0.25,
                                                min_lr=0.000001)


    optimizer = Adam(learning_rate=0.5e-4, beta_1=0.9, beta_2=0.999, amsgrad=False)


    base_model=ResNet50(include_top=False, weights= 'imagenet',  input_shape=(150,150,3), pooling='avg')
    base_model.trainable = False

    x = Dense(512, activation='relu')(base_model.output)
    x = Dropout(0.5)(x)
    x = Dense(6, activation='softmax')(x)

    transfer_model = Model(base_model.input, x) 
    transfer_model.compile(optimizer =optimizer, 
                  loss = 'categorical_crossentropy', 
                  metrics = ['accuracy'])


    history = transfer_model.fit_generator(
        train_gen, 
        steps_per_epoch  = 256, 
        validation_data  = test_gen,
        validation_steps = 256,
        epochs = 1, 
        verbose = 1,
        callbacks = [es, learning_rate_reduction]
    )


    transfer_model.save("res_model.model")


    def get_images(directory):
        Images = []
        Labels = [] 
        label = 0

        for labels in os.listdir(directory): 
            if labels == 'glacier':
                label = 2
            elif labels == 'sea':
                label = 4
            elif labels == 'buildings':
                label = 0
            elif labels == 'forest':
                label = 1
            elif labels == 'street':
                label = 5
            elif labels == 'mountain':
                label = 3

            for image_file in os.listdir(directory+labels):
                image = cv2.imread(directory+labels+r'/'+image_file)
                image = cv2.resize(image,(150,150))
                Images.append(image)
                Labels.append(label)

        return Images,Labels


    test_images,test_labels = get_images('../input/intel-image-classification/seg_test/seg_test/')
    test_images = np.array(test_images)
    test_labels = np.array(test_labels)
    test_labels = to_categorical(test_labels)


    return transfer_model.evaluate(test_images, test_labels)
