import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, BatchNormalization, LeakyReLU, Flatten, Activation, MaxPool2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, RMSprop
from keras.callbacks import EarlyStopping,ReduceLROnPlateau,LearningRateScheduler

train_dir = '../input/intel-image-classification/seg_train/seg_train/'
test_dir = '../input/intel-image-classification/seg_test/seg_test/'

data_gen = ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=10,
    zoom_range = 0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=False,
    vertical_flip=False)

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

def build_model():
    
    model = Sequential()
    
    
    model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(150,150,3)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=3, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2,2)))
    
    model.add(Conv2D(128, kernel_size=3, activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, kernel_size=3, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2,2)))
    
    model.add(Conv2D(256, kernel_size=3, activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, kernel_size=3, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2,2)))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(6, activation='softmax'))

    
    return model

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

model = build_model()
model.compile(
    loss='binary_crossentropy',
    optimizer=optimizer,
    metrics=['accuracy']
)

history = model.fit_generator(
    train_gen, 
    steps_per_epoch  = 256, 
    validation_data  = test_gen,
    validation_steps = 256,
    epochs = 30, 
    verbose = 1,
    callbacks = [es, learning_rate_reduction]
)

model.save("high_accuracy_complex.h5")

import os
import cv2

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

from keras.utils import to_categorical

test_images,test_labels = get_images('../input/intel-image-classification/seg_test/seg_test/')
test_images = np.array(test_images)
test_labels = np.array(test_labels)
test_labels = to_categorical(test_labels)

model.evaluate(test_images, test_labels)