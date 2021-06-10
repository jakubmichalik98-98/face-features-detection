import cv2 as cv
from basic_functions import read_image, show_images
from luminance_correction import luminance_correction
from face_candidates import create_binary_mask, fit_ellipse
from copy import deepcopy
from mouth_candidates import create_mouth_map_Fg
from eye_candidates import create_eye_map_y, create_eye_map_c, final_c_y_mask

if __name__ == '__main__':
    img1 = read_image("data/29-1m.bmp")
    img1_rgb = cv.cvtColor(img1, cv.COLOR_BGR2RGB)
    img1_to_show = deepcopy(img1_rgb)
    luminance_correction(img1_rgb)
    img1_corrected_to_show = deepcopy(img1_rgb)
    mask = create_binary_mask(img1_rgb)
    ellipse_img = fit_ellipse(mask, img1_rgb)
    ellipse_img_to_show = deepcopy(ellipse_img)
    ellipse_img_YCrCb = cv.cvtColor(ellipse_img, cv.COLOR_RGB2YCrCb)

    c_eye_mask = create_eye_map_c(ellipse_img_YCrCb)
    y_eye_mask = create_eye_map_y(ellipse_img_YCrCb)
    c_y_eye_mask = final_c_y_mask(c_eye_mask, y_eye_mask)

    mouth_map = create_mouth_map_Fg(ellipse_img_YCrCb)

    show_images([img1_to_show, img1_corrected_to_show, mask, ellipse_img_to_show, c_eye_mask, y_eye_mask, c_y_eye_mask, mouth_map], 4, 2)


