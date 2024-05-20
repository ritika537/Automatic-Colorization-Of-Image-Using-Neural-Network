#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Input, BatchNormalization, LeakyReLU, concatenate
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import os
import time


# In[2]:


np.random.seed(1)
tf.random.set_seed(123)


# In[3]:


def ExtractInput(path):
    X_img = []
    y_img = []
    for imageDir in os.listdir(path):
        try:
            img = cv2.imread(os.path.join(path, imageDir))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

            img = img.astype(np.float32)
            img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
            img_lab_rs = cv2.resize(img_lab, (WIDTH, HEIGHT))
            img_l = img_lab_rs[:, :, 0]
            img_ab = img_lab_rs[:, :, 1:] / 128
            X_img.append(img_l[..., np.newaxis])  # Add channel dimension
            y_img.append(img_ab)
        except:
            pass
    X_img = np.array(X_img)
    y_img = np.array(y_img)

    return X_img, y_img


# In[4]:


HEIGHT = 224
WIDTH = 224
ImagePath = "D:/dataset_updated/training_set/painting"


# In[5]:


X_, y_ = ExtractInput(ImagePath)


# In[6]:


def InstantiateModel(in_):
    model_ = Conv2D(16, (3, 3), padding='same', strides=1)(in_)
    model_ = LeakyReLU()(model_)
    model_ = Conv2D(32, (3, 3), padding='same', strides=1)(model_)
    model_ = LeakyReLU()(model_)
    model_ = BatchNormalization()(model_)
    model_ = MaxPooling2D(pool_size=(2, 2), padding='same')(model_)

    model_ = Conv2D(64, (3, 3), padding='same', strides=1)(model_)
    model_ = LeakyReLU()(model_)
    model_ = BatchNormalization()(model_)
    model_ = MaxPooling2D(pool_size=(2, 2), padding='same')(model_)

    model_ = Conv2D(128, (3, 3), padding='same', strides=1)(model_)
    model_ = LeakyReLU()(model_)
    model_ = BatchNormalization()(model_)

    model_ = Conv2D(256, (3, 3), padding='same', strides=1)(model_)
    model_ = LeakyReLU()(model_)
    model_ = BatchNormalization()(model_)

    model_ = UpSampling2D((2, 2))(model_)
    model_ = Conv2D(128, (3, 3), padding='same', strides=1)(model_)
    model_ = LeakyReLU()(model_)
    model_ = BatchNormalization()(model_)

    model_ = UpSampling2D((2, 2))(model_)
    model_ = Conv2D(64, (3, 3), padding='same', strides=1)(model_)
    model_ = LeakyReLU()(model_)

    concat_ = concatenate([model_, in_])

    model_ = Conv2D(64, (3, 3), padding='same', strides=1)(concat_)
    model_ = LeakyReLU()(model_)
    model_ = BatchNormalization()(model_)

    model_ = Conv2D(32, (3, 3), padding='same', strides=1)(model_)
    model_ = LeakyReLU()(model_)

    model_ = Conv2D(2, (3, 3), activation='tanh', padding='same', strides=1)(model_)

    return model_


# In[7]:


Input_Sample = Input(shape=(HEIGHT, WIDTH, 1))


# In[8]:


Output_ = InstantiateModel(Input_Sample)
Model_Colourization = Model(inputs=Input_Sample, outputs=Output_)


# In[9]:


LEARNING_RATE = 0.001
Model_Colourization.compile(optimizer=Adam(lr=LEARNING_RATE), loss='mean_squared_error')


# In[10]:


datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')


# In[11]:


batch_size = 32


# In[12]:


train_generator = datagen.flow(X_, y_, batch_size=batch_size)


# In[ ]:




