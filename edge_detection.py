import cv2 as cv
from basic_functions import read_image, show_images
from luminance_correction import luminance_correction
from eye_candidates import final_c_y_mask, create_eye_map_y, create_eye_map_c
import matplotlib.pyplot as plt
from face_candidates import fit_ellipse, create_binary_mask

if __name__ == '__main__':
    img = read_image("D:/Wykrywanie twarzy/face_data/data/01-1m.bmp")
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    luminance_correction(img_rgb)
    mask = create_binary_mask(img_rgb)
    ellipse_img = fit_ellipse(mask, img_rgb)
    imgYCrCb = cv.cvtColor(ellipse_img, cv.COLOR_RGB2YCrCb)

    eye_map_c = create_eye_map_c(imgYCrCb)
    eye_map_y = create_eye_map_y(imgYCrCb)
    eye_mask = final_c_y_mask(eye_map_c, eye_map_y)
