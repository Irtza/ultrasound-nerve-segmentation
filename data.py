## Modified by github.com/irtza 
## to Work with .png images and .png masks  on 3 channel Tensorflow ordered Data. 

from __future__ import print_function

import os
import numpy as np

from skimage.io import imsave, imread

data_path = '../splitdata/'

# Original Dimensions of the images. 
image_rows = 640
image_cols = 640
img_channels = 3
## Load Everything to Memory and Save as Numpy Arrays .npy. 


def create_train_data():
    '''
        Load same file name images Simulataneuosly from Images and Mask folders. 
        Save the Images to numpy for quick loading
    '''
    train_data_path = os.path.join(data_path, 'trainX')
    train_label_path = os.path.join(data_path , 'trainY')

    filelist = os.listdir(train_data_path)
    total = len(filelist)

    imgs      = np.ndarray((total, image_rows, image_cols , img_channels), dtype=np.uint8)
    imgs_mask = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)

    i = 0
    print('-'*30)
    print('Creating training images...')
    print('-'*30)

    for image_name in filelist:
        img      = imread(os.path.join(train_data_path, image_name), as_grey=False)[:,:,:3]  # rgb Collor Image
        img_mask = imread(os.path.join(train_label_path, image_name), as_grey=True)

        img = np.array([img])
        img_mask = np.array([img_mask])

        imgs[i] = img
        imgs_mask[i] = img_mask

        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1

    print('-'*30)
    print('Verifying Shapes of the Arrays..')
    print(imgs.shape , imgs_mask.shape)
    print('-'*30)

    np.save('imgs_train.npy', imgs)
    np.save('imgs_mask_train.npy', imgs_mask)
    print('Saving to .npy files done.')



def create_test_data():
    test_data_path = os.path.join(data_path, 'testX')
    test_label_path = os.path.join(data_path, 'testY')

    filelist = os.listdir(test_data_path)
    total = len(filelist)

    imgs      = np.ndarray((total, image_rows, image_cols , img_channels), dtype=np.uint8)
    imgs_mask = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)
    imgs_id =   []

    i = 0
    print('-'*30)
    print('Creating test images...')
    print('-'*30)

    for image_name in filelist:

        img = imread(os.path.join(test_data_path, image_name), as_grey=False)[:,:,:3]
        img_mask = imread(os.path.join(test_label_path, image_name), as_grey=True)
        
        img = np.array([img])
        img_mask = np.array([img_mask])

        # Load into Arrays
        imgs_id.append(image_name)

        imgs[i] = img
        imgs_mask[i] = img_mask

        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1

    print('Loading done.')

    imgs_id = np.array(imgs_id)

    np.save('imgs_test.npy', imgs)
    np.save('imgs_id_test.npy', imgs_id)
    np.save('imgs_mask_test_GT.npy', imgs_mask)
    print('Saving to .npy files done.')



def load_train_data():
    imgs_train = np.load('imgs_train.npy')
    imgs_mask_train = np.load('imgs_mask_train.npy')
    return imgs_train, imgs_mask_train

def load_test_data():
    imgs_test = np.load('imgs_test.npy')
    imgs_mask_test = np.load('imgs_mask_test.npy')
    imgs_id = np.load('imgs_id_test.npy')
    return imgs_test, imgs_id , imgs_mask_test

if __name__ == '__main__':
    create_train_data()
    create_test_data()


    print('-'*30)
    print('Done ! ')
