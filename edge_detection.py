import cv2 as cv
from basic_functions import read_image, show_images
from luminance_correction import luminance_correction
from eye_candidates import final_c_y_mask, create_eye_map_y, create_eye_map_c
import matplotlib.pyplot as plt
from face_candidates import fit_ellipse, create_binary_mask
import numpy as np

if __name__ == '__main__':
    img = read_image("data/18-1m.bmp")
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    luminance_correction(img_rgb)
    mask = create_binary_mask(img_rgb)
    ellipse_img = fit_ellipse(mask, img_rgb)
    imgYCrCb = cv.cvtColor(ellipse_img, cv.COLOR_RGB2YCrCb)

    eye_map_c = create_eye_map_c(imgYCrCb)
    eye_map_y = create_eye_map_y(imgYCrCb)
    eye_mask = final_c_y_mask(eye_map_c, eye_map_y)

    layer = eye_mask.copy()
    gaussian_pyramid_layers = [layer]
    for i in range(6):
        layer = cv.pyrDown(layer)
        gaussian_pyramid_layers.append(layer)

    laplacian_layer = gaussian_pyramid_layers[5]
    laplacian_pyramid_layers = [laplacian_layer]
    for i in range(5, 0, -1):
        gaussian_extended = cv.pyrUp(gaussian_pyramid_layers[i])
        laplacian = cv.subtract(gaussian_pyramid_layers[i-1], gaussian_extended)
        laplacian_pyramid_layers.append(laplacian)
    # show_images([laplacian_pyramid_layers[-1], eye_mask], 2, 1)

    pyramid_img = laplacian_pyramid_layers[-1]
    pyramid_img_rgb = cv.cvtColor(pyramid_img, cv.COLOR_YCrCb2RGB)
    pyramid_img_gray = cv.cvtColor(pyramid_img_rgb, cv.COLOR_RGB2GRAY)
    ret, th = cv.threshold(pyramid_img_gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    kernel = np.ones((5, 5), np.uint8)
    closing = cv.morphologyEx(th, cv.MORPH_CLOSE, kernel)
    # show_images([eye_mask], 1, 1)
    plt.subplot(1, 1, 1)
    plt.imshow(closing, cmap="gray")
    plt.show()



