import cv2 as cv
from basic_functions import read_image, show_images
from luminance_correction import luminance_correction
from face_candidates import create_binary_mask, fit_ellipse
from copy import deepcopy

if __name__ == '__main__':
    img1 = read_image("D:/Wykrywanie twarzy/face_data/data/29-1m.bmp")
    img1_rgb_raw = cv.cvtColor(img1, cv.COLOR_BGR2RGB)
    img1_rgb = cv.cvtColor(img1, cv.COLOR_BGR2RGB)
    corrected_image = luminance_correction(img1_rgb)
    img2 = deepcopy(corrected_image)
    mask = create_binary_mask(img2)
    ellipse_img = fit_ellipse(mask, img2)
    show_images([img1_rgb_raw, corrected_image, img2], 3, 1)


