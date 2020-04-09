# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 11:58:00 2020

@author: gev
"""

import random
import warnings
import glob
import matplotlib.pyplot as plt
import cv2

import numpy as np

from keras.models import Model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint

from skimage.io import imread, imshow, concatenate_images, imsave
from skimage.util import invert
from skimage.transform import resize
import tensorflow as tf

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 1

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
seed = 42
random.seed = seed
np.random.seed = seed

def img_processing(img, img_size, mask):
    
    if img.shape[0] == img.shape[1]:
        resized_shape = (img_size, img_size)
        offset = (0, 0)
    elif img.shape[0] > img.shape[1]:
        resized_shape = (img_size, round(img_size * img.shape[1] / img.shape[0]))
        offset = (0, (img_size - resized_shape[1]) // 2)
    else:
        resized_shape = (round(img_size * img.shape[0] / img.shape[1]), img_size)
        offset = ((img_size - resized_shape[0]) // 2, 0)
    
    resized_shape1 = (resized_shape[1], resized_shape[0])
    
    if (mask == 1):
        img_resized = cv2.resize(img.astype(np.uint8), resized_shape1).astype(np.bool)
        img_padded = np.zeros((img_size, img_size), dtype=np.bool)
    else:
        img_resized = cv2.resize(img, resized_shape1).astype(np.uint8)
        img_padded = np.zeros((img_size, img_size), dtype=np.uint8)
    
    img_padded[offset[0] : (offset[0] + resized_shape[0]), offset[1] : (offset[1] + resized_shape[1])] = img_resized 
    return img_padded


X_train = np.zeros((155, IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.uint8)
Y_train = np.zeros((155, IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
n = 0

for q, file_path in enumerate(glob.glob('E:/Projects/ML/segmentation/NLM-MontgomeryCXRSet/MontgomerySet/CXR_png/*.png')):
    print(n)
    img = imread(file_path)
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    img = np.expand_dims(resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant',  preserve_range=True), axis=-1)
    X_train[n] = img
    left_mask_name = 'e:/Projects/ML/segmentation/NLM-MontgomeryCXRSet/MontgomerySet/ManualMask/leftMask/'+file_path[-17:]
    right_mask_name = 'e:/Projects/ML/segmentation/NLM-MontgomeryCXRSet/MontgomerySet/ManualMask/rightMask/'+file_path[-17:]
    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
    for mask_name in [left_mask_name, right_mask_name]:
        mask_ = imread(mask_name)
        mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant',  preserve_range=True), axis=-1)
        mask = np.maximum(mask, mask_)
    Y_train[n] = mask
    n += 1

for q, file_path in enumerate(glob.glob('e:/Shared/Images/AI/1/8/*.tif')):
    print(n)
    img = imread(file_path)
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    img = np.expand_dims(resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant',  preserve_range=True), axis=-1)
    X_train[n] = img
    
    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
    Y_train[n] = mask
    n += 1

for q, file_path in enumerate(glob.glob('E:/Projects/ML/segmentation/NLM-MontgomeryCXRSet/Lungs-jpeg/Lungs-jpeg/ds/masks_human/*.png')):
    print(n)
    img = imread(file_path)
    shape = img.shape
    img1 = img[:, 0: np.uint32(img.shape[1]/2), :]
    
    img2 = img[:, np.uint32(img.shape[1]/2):, :]
    img3 = img2[:,:,0] - img2[:,:,1]
    img3[img3 > 0] = 1
    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
    mask_ = np.expand_dims(resize(img3, (IMG_HEIGHT, IMG_WIDTH), mode='constant',  preserve_range=True), axis=-1)
    mask = np.maximum(mask, mask_)
    
    img = resize(img1[:,:,0], (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    img = np.expand_dims(resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant',  preserve_range=True), axis=-1)
    X_train[n] = img
    Y_train[n] = mask
    n += 1

X = X_train
Y = Y_train

inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
s = Lambda(lambda x: x / 255) (inputs)

c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (s)
c1 = Dropout(0.1) (c1)
c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c1)
p1 = MaxPooling2D((2, 2)) (c1)

c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p1)
c2 = Dropout(0.1) (c2)
c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c2)
p2 = MaxPooling2D((2, 2)) (c2)

c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p2)
c3 = Dropout(0.2) (c3)
c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c3)
p3 = MaxPooling2D((2, 2)) (c3)

c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p3)
c4 = Dropout(0.2) (c4)
c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c4)
p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p4)
c5 = Dropout(0.3) (c5)
c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c5)

u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c5)
u6 = concatenate([u6, c4])
c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u6)
c6 = Dropout(0.2) (c6)
c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c6)

u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6)
u7 = concatenate([u7, c3])
c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u7)
c7 = Dropout(0.2) (c7)
c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c7)

u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7)
u8 = concatenate([u8, c2])
c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u8)
c8 = Dropout(0.1) (c8)
c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c8)

u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)
u9 = concatenate([u9, c1], axis=3)
c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u9)
c9 = Dropout(0.1) (c9)
c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c9)

outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)

model = Model(inputs=[inputs], outputs=[outputs])
from keras import optimizers # sgdbinary_crossentropy
sgd = optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer='adam', loss='mean_squared_error', metrics=[tf.keras.metrics.MeanIoU(num_classes=2)])
model.summary()

earlystopper = EarlyStopping(patience=5, verbose=1)
checkpointer = ModelCheckpoint('model.h5', verbose=1, save_best_only=True)
results = model.fit(X, Y, validation_split=0.2, batch_size=32, epochs=200, callbacks=[earlystopper, checkpointer])

preds_test = model.predict(X_test, verbose=1)
preds_test_t = (preds_test > 0.5).astype(np.uint8)
preds_test_upsampled = []

preds_test_upsampled = []
for i in range(len(preds_test)):
    preds_test_upsampled.append(resize(np.squeeze(preds_test[i]), 
                                       (X_test.shape[1], X_test.shape[2]), 
                                       mode='constant', preserve_range=True))
imshow(X_test[0])
plt.show()
imshow(np.squeeze(preds_test_upsampled[0]))
plt.show()
import matplotlib
imgs = np.zeros((22, IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.uint8)
n = 0

for q, file_path in enumerate(glob.glob('e:/Shared/Images/AI/Agfa/8/*.tif')):    
    img = invert(imread(file_path))
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    img = np.expand_dims(resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant',  preserve_range=True), axis=-1)
    imgs[n] = img    
    matplotlib.image.imsave("e:/Shared/Images/AI/Agfa/test/"+str(n)+".png", resize(np.squeeze(imgs[n]), (img.shape[0], img.shape[1]), mode='constant', preserve_range=True))
    n += 1

predict = model.predict(imgs, verbose=1)
#predict = (predict > 0.7).astype(np.uint8)

for i in range(22):
    matplotlib.image.imsave("e:/Shared/Images/AI/Agfa/predict/"+str(i)+".png", resize(np.squeeze(predict[i]), (img.shape[0], img.shape[1]), mode='constant', preserve_range=True))
    #imsave("e:/Shared/Images/AI/Telekord/predict/"+str(i)+".png", img_as_uint(resize(np.squeeze(predict[i]), (img.shape[0], img.shape[1])))) 