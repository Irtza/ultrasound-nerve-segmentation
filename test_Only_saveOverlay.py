from __future__ import print_function

import os
from skimage.transform import resize
from skimage.io import imsave
from skimage import color
import numpy as np

from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K
import keras
from data import load_train_data, load_test_data

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

img_rows = 160
img_cols = 160
img_channels = 3

smooth = 1.

## Train on RGB satellite Image. 


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def get_unet():
    inputs = Input((img_rows, img_cols, img_channels))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])
    return model

def overlay_on_gray(oimg,overlayGREEN=None,overlayRED=None,overlayBLUE=None,alpha = 0.6):
    rows, cols = oimg.shape
    color_mask = np.zeros((rows, cols, 3))
    if overlayGREEN is not None:
        color_mask[:,:,1] = overlayGREEN
    if overlayRED  is not None:
        color_mask[:,:,2] = overlayRED
    if overlayBLUE is not None:
        color_mask[:,:,0] = overlayBLUE
    
    img_color = np.dstack((oimg, oimg, oimg))

    img_hsv = color.rgb2hsv(img_color)
    color_mask_hsv = color.rgb2hsv(color_mask)
    
    img_hsv[..., 0] = color_mask_hsv[..., 0]
    img_hsv[..., 1] = color_mask_hsv[..., 1] * alpha
    img_masked = color.hsv2rgb(img_hsv)

    return img_masked,color_mask




def preprocess(imgs):
    if imgs.ndim == 4: 
        imgs_p = np.ndarray((imgs.shape[0], img_rows, img_cols, img_channels), dtype=np.uint8)
        for i in range(imgs.shape[0]):
            imgs_p[i] = resize(imgs[i], (img_cols, img_rows,img_channels), preserve_range=True)

    else :
        imgs_p = np.ndarray((imgs.shape[0], img_rows, img_cols), dtype=np.uint8)
        for i in range(imgs.shape[0]):
            imgs_p[i] = resize(imgs[i], (img_cols, img_rows), preserve_range=True)

        imgs_p = imgs_p[..., np.newaxis]

    return imgs_p


def load_and_predict():
    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)
    imgs_train, imgs_mask_train = load_train_data()

    ## Image Resizing
    imgs_train = preprocess(imgs_train)
    imgs_mask_train = preprocess(imgs_mask_train)

    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    model = get_unet()

    print('-'*30)
    print('Loading and preprocessing test data... NOT LOADING LABELS')
    print('-'*30)
    imgs_test, imgs_id_test , _ = load_test_data()
    o_imgs_test = imgs_test

    imgs_test = preprocess(imgs_test)

    imgs_test = imgs_test.astype('float32')
    imgs_test -= mean
    imgs_test /= std

    print('-'*30)
    print('Loading saved weights...')
    print('-'*30)
    model.load_weights('weights.h5')

    print('-'*30)
    print('Predicting masks on test data...')
    print('-'*30)
    imgs_mask_test = model.predict(imgs_test, verbose=1)
    np.save('imgs_mask_test_PRED.npy', imgs_mask_test)
    print('-' * 30)
    print('Saving predicted masks and Overlays to ./preds/ and ./overlay/...')
    print('-' * 30)

    pred_dir = 'preds'
    overlay_dir = 'overlay'

    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
    if not os.path.exists(overlay_dir):
        os.mkdir(overlay_dir)

    for image, image_id , o_test_img in zip(imgs_mask_test, imgs_id_test , o_imgs_test):
        image = (image[:, :, 0] * 255.).astype(np.uint8)
        imsave(os.path.join(pred_dir, image_id), image)
        # Use original dim Image and Saving Overlay onto
        overlay_img, _ = overlay_on_gray(color.rgb2gray(o_test_img),overlayGREEN=image.astype(bool))
        imsave(os.path.join(overlay_dir, image_id), overlay_img)

if __name__ == '__main__':

    print('-' * 30)
    print('Training ... and Predicting')
    load_and_predict()

