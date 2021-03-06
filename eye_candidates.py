import cv2 as cv
import numpy as np
from basic_functions import read_image, show_images
from luminance_correction import luminance_correction
from copy import deepcopy


def create_eye_map_c(imgYCrCb):
    """
    Function responsible for calculate eye map from chrominance components
    :param imgYCrCb: image in YCrCb space
    :return: Eye map from chrominance components
    """
    processed_img = deepcopy(imgYCrCb)
    h = imgYCrCb.shape[0]
    w = imgYCrCb.shape[1]
    cb_sqr = np.zeros(imgYCrCb.shape)
    cb_over_cr = np.zeros(imgYCrCb.shape)
    cr_neg_sqr = np.zeros(imgYCrCb.shape)
    for i in range(h):
        for j in range(w):
            cb_sqr[i, j] = imgYCrCb[i, j, 2] ** 2
            cb_over_cr[i, j] = imgYCrCb[i, j, 2] / imgYCrCb[i, j, 1]
            cr_neg_sqr[i, j] = (255 - imgYCrCb[i, j, 1]) ** 2
    cv.normalize(cb_sqr, cb_sqr, 0, 255, cv.NORM_MINMAX)
    cv.normalize(cb_over_cr, cb_over_cr, 0, 255, cv.NORM_MINMAX)
    cv.normalize(cr_neg_sqr, cr_neg_sqr, 0, 255, cv.NORM_MINMAX)

    for i in range(h):
        for j in range(w):
            processed_img[i, j] = (cb_sqr[i, j] + cr_neg_sqr[i, j] +
                                   cb_over_cr[i, j]) / 3
    processed_img[:, :, 0] = cv.equalizeHist(processed_img[:, :, 0])
    processed_img[:, :, 1] = cv.equalizeHist(processed_img[:, :, 1])
    processed_img[:, :, 2] = cv.equalizeHist(processed_img[:, :, 2])

    return processed_img


def create_eye_map_y(imgYCrCb):
    """
    Function responsible for creating eye map from luminance component.
    Not working yet probably due to structurAing element.
    :param imgYCrCb: image in YCrCb space
    :return: Eye map from luminance component
    """
    processed_img = deepcopy(imgYCrCb)
    str_el = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    y_dilation = cv.dilate(processed_img[:, :, 0], str_el)
    y_erosion = cv.erode(processed_img[:, :, 0], str_el)
    h = imgYCrCb.shape[0]
    w = imgYCrCb.shape[1]
    for i in range(h):
        for j in range(w):
            processed_img[i, j] = 10 * (y_dilation[i, j] / (1 + y_erosion[i, j] / 10))
    return processed_img

def final_c_y_mask(eye_map_c, eye_map_y):
    """
    Function responsible for joining two eye maps together
    :param eye_map_c: C eye map, result of function create_eye_map_c
    :param eye_map_c: Y eye map, result of function create_eye_map_y
    :return: Joined eye maps
    """
    return cv.bitwise_and(eye_map_c, eye_map_y)

if __name__ == '__main__':
    """
    Only for testing in final version it must be moved to main.py file
    """
    img = read_image("data/07-1m.bmp")
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    luminance_correction(img_rgb)
    imgYCrCb = cv.cvtColor(img_rgb, cv.COLOR_RGB2YCrCb)
    eye_map_c = create_eye_map_c(imgYCrCb)
    eye_map_y = create_eye_map_y(imgYCrCb)
    final_mask = final_c_y_mask(eye_map_c, eye_map_y)
    show_images([imgYCrCb, eye_map_c, eye_map_y, final_mask], 4, 1)
