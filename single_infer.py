import os
from argparse import ArgumentParser
import time

import numpy
import numpy as np
import cv2
import matplotlib
from mmcv.cnn.utils import revert_sync_batchnorm
import matplotlib.pyplot as plt
from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot, show_result_pyplot_new, \
    inference_segmentor_logit
from skimage import util
import cv2
import numpy as np
import matplotlib.pyplot as plt


def add_gaussian_noise(image, mean=0, sigma=15):
    gaussian_noise = np.random.normal(mean, sigma, image.shape)
    noisy_image = image + gaussian_noise
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

    return noisy_image


IMG_WIDTH, IMG_HEIGHT = 1280, 1280


def crop_image_from_gray(img, tol=7):
    """
    Applies masks to the orignal image and
    returns the a preprocessed image with
    3 channels

    :param img: A NumPy Array that will be cropped
    :param tol: The tolerance used for masking

    :return: A NumPy array containing the cropped image
    """
    # If for some reason we only have two channels
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    # If we have a normal RGB images
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        mask = gray_img > tol

        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]

        if check_shape == 0:  # image is too dark so that we crop out everything,
            return img  # return original image
        else:

            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)
        return img



import cv2
from skimage import util
import numpy as np


def preprocess_image(image_path, sigmaX=15,sigma=1, IMG_WIDTH=1280, IMG_HEIGHT=1280):
    # Load image
    image = cv2.imread(image_path)
    # Convert color
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
    # gaussian_noise = np.random.normal(0, 1, image.shape).astype('uint8')
    # image = cv2.add(image, gaussian_noise)
    # Convert back to BGR for output

    img_ori = add_gaussian_noise(image, mean=0, sigma=sigma)
    img_ori = cv2.cvtColor(img_ori, cv2.COLOR_RGB2BGR)
    # Apply weighted addition with Gaussian blur
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)
    image = add_gaussian_noise(image, mean=0, sigma=sigma)
    return image, img_ori


def count_elements(matrix):
    counts = {}
    for row in matrix:
        for element in row:
            if element in counts:
                counts[element] += 1
            else:
                counts[element] = 1
    return counts


config_file = ''
checkpoint_file = ''
img_dir = './data/ddr_/test'
model = init_segmentor(config_file, checkpoint_file, device='cpu')
model = revert_sync_batchnorm(model)
l = []
name_list = []
PALETTE = [[0, 0, 0], [255, 0, 0], [255, 255, 0], [255, 255, 255], [0, 255, 0]]
img, img_ori = preprocess_image('./data/68475.jpg',sigma=10)
cv2.imwrite('./data/img.jpg',img)
cv2.imwrite('./data/img_ori.jpg',img_ori)
result = inference_segmentor(model, img)
mask = numpy.zeros((1280, 1280, 3), dtype=numpy.int8)
show_result_pyplot_new(model, img_ori, result[0], opacity=0.3, show=False,
                       out_file=f'./data/persuade2_l'+'.png',
                       palette=[[0, 0, 0], [255, 0, 0], [255, 255, 0], [0, 255, 0], [0, 0, 255]],
                       )
