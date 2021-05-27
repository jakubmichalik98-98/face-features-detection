import cv2 as cv
from basic_functions import read_image, show_images
from luminance_correction import luminance_correction

if __name__ == '__main__':
    img1 = read_image("data/01-1m.bmp")
    img1_rgb_raw = cv.cvtColor(img1, cv.COLOR_BGR2RGB)
    img1_rgb = cv.cvtColor(img1, cv.COLOR_BGR2RGB)
    corrected_image = luminance_correction(img1_rgb)
    show_images([img1_rgb_raw, corrected_image], 2, 1)

