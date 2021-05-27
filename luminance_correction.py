import cv2 as cv
import numpy as np
from basic_functions import read_image, show_images


def find_luma_value(img):
    """
    Function responsible for calculate luma value of each pixel
    :param img: Read image
    :return: Array with luma values
    """
    luma_values_array = np.ndarray(shape=(480, 640), dtype=float)
    h = img.shape[0]
    w = img.shape[1]
    for i in range(h):
        for j in range(w):
            luma_values_array[i, j] = img[i, j, 0] * 0.2126 + img[i, j, 1] * 0.7152 + img[i, j, 2] * 0.0722
    return luma_values_array


def find_range_of_luminance(luminance_values):
    """
    Function responsible for finding range of luminance values
    :param luminance_values: array with luma values
    :return: range of luma values from array
    """
    return [luminance_values.min(), luminance_values.max()]


def find_top_5_percent_pixels_of_luminance_value(luminance_values, max_luminance):
    """
    Function responsible for finding pixels with over 95% of max luminance value
    :param luminance_values: array with luma values
    :param max_luminance: Max luminance value from array
    :return: pixel indexes with luma values over 95% of max luminance value
    """
    searched_pixel_indexes = []
    top_five_percent = max_luminance * 0.95
    h = luminance_values.shape[0]
    w = luminance_values.shape[1]
    for i in range(h):
        for j in range(w):
            if luminance_values[i, j] >= top_five_percent:
                searched_pixel_indexes.append([i, j])
    return searched_pixel_indexes


def calculate_average_of_pixels_from_top_5(top_5_pixels, img):
    """
    Function responsible for calculating average from pixels with luma values over 95% of max luminance
    value
    :param top_5_pixels: indexes of pixels with luma values over 95% of max luminance
    :param img: read image
    :return: Reference pixel values
    """
    r = 0
    g = 0
    b = 0
    for pixel in top_5_pixels:
        r += img[pixel[0], pixel[1], 0]
        g += img[pixel[0], pixel[1], 1]
        b += img[pixel[0], pixel[1], 2]
    return [r / len(top_5_pixels), g / len(top_5_pixels), b / len(top_5_pixels)]


def normalize_pixels(img, reference_color):
    """
    Function responsible for normalize pixels to the reference color
    :param img: Read image
    :param reference_color: Reference color
    :return: Image with pixels normalized to reference color
    """
    h = img.shape[0]
    w = img.shape[1]
    for i in range(h):
        for j in range(w):
            img[i, j] = [(reference_color[0] * img[i, j, 0]) // 255,
                         (reference_color[1] * img[i, j, 1]) // 255,
                         (reference_color[2] * img[i, j, 2]) // 255]
    return img


def compare_images(img1, img2):
    """
    Optional function to compare values from raw image and luminance corrected image
    :param img1: First image
    :param img2: Second image
    :return: Nothing it print difference between image pixel values
    """
    h = img1.shape[0]
    w = img1.shape[1]
    for i in range(h):
        for j in range(w):
            if img1[i, j, 0] != img2[i, j, 0]:
                print(img1[i, j, 0] - img2[i, j, 0])
            if img1[i, j, 1] != img2[i, j, 1]:
                print(img1[i, j, 1] - img2[i, j, 1])
            if img1[i, j, 2] != img2[i, j, 2]:
                print(img1[i, j, 2] - img2[i, j, 2])


def luminance_correction(img1_rgb):
    """
    Function responsible for return luminance corrected image
    :param img1_rgb: input image
    :return: luminanced corrected image
    """
    luma_values = find_luma_value(img1_rgb)
    luma_range = find_range_of_luminance(luma_values)
    top_5 = find_top_5_percent_pixels_of_luminance_value(luma_values, luma_range[1])
    reference_white_color = calculate_average_of_pixels_from_top_5(top_5, img1_rgb)
    luma_corrected_image = normalize_pixels(img1_rgb, reference_white_color)
    return luma_corrected_image


