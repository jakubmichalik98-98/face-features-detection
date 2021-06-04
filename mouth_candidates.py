import cv2 as cv
from basic_functions import read_image, show_images
from luminance_correction import luminance_correction
from copy import deepcopy
import numpy as np


def create_mouth_map_Fg(imgYCrCb):
    """
    Function responsible for calculate mouth map from chrominance components
    :param imgYCrCb: image in YCrCb space
    :return: Mouth map calculated from chrominance components based on formula (3)
    """
    processed_img = deepcopy(imgYCrCb)
    h = imgYCrCb.shape[0]
    w = imgYCrCb.shape[1]
    eta = calculate_eta(imgYCrCb)
    cr_sqr = np.zeros(imgYCrCb.shape)
    cr_over_cb = np.zeros(imgYCrCb.shape)

    for i in range(h):
        for j in range(w):
            cr_sqr[i,j] = imgYCrCb[i, j, 1] ** 2
            cr_over_cb[i, j] = imgYCrCb[i, j, 1]  / imgYCrCb[i, j, 2]
    
    cv.normalize(cr_sqr, cr_sqr, 0, 255, cv.NORM_MINMAX)
    cv.normalize(cr_over_cb, cr_over_cb, 0, 255, cv.NORM_MINMAX)

    for i in range(h):
        for j in range(w):
            processed_img[i, j] = cr_sqr[i, j]  * (cr_sqr[i, j] - eta * cr_over_cb[i, j]) ** 2

    return processed_img

def calculate_eta(imgYCrCb):
    """
    Function responsible for calculating constant eta from chrominance components of the input image
    :param imgYCrCb: image in YCrCb space
    :return: Constant eta calculated from chrominance components based on formula (3)
    """
    h = imgYCrCb.shape[0]
    w = imgYCrCb.shape[1]
    n = (h + 1) * (w + 1) # "Fg is the face mask with n points"
    cr_sqr = np.zeros(imgYCrCb.shape)
    cr_over_cb = np.zeros(imgYCrCb.shape)

    for i in range(h):
        for j in range(w):
            cr_sqr[i, j] = imgYCrCb[i, j, 1] ** 2
            cr_over_cb[i, j] = imgYCrCb[i, j, 1] / imgYCrCb[i, j, 2]

    cv.normalize(cr_sqr, cr_sqr, 0, 255, cv.NORM_MINMAX)
    cv.normalize(cr_over_cb, cr_over_cb, 0, 255, cv.NORM_MINMAX)
    
    cr_sqr_sum = np.sum(cr_sqr)
    cr_over_cb_sum = np.sum(cr_over_cb)

    eta = 0.95 * ( cr_sqr_sum / n ) / ( cr_over_cb_sum / n )
    return eta

if __name__ == '__main__':
    """
    Only for testing in final version it must be moved to main.py file
    """
    img = read_image("D:/AAA/Studia/Informatics/1semestr/CVaPR/Project/data/26-5m.jpg")
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    luminance_correction(img_rgb)
    imgYCrCb = cv.cvtColor(img_rgb, cv.COLOR_RGB2YCrCb)
    mouth_map_Fg = create_mouth_map_Fg(imgYCrCb)
    show_images([imgYCrCb, mouth_map_Fg ], 2, 1)
