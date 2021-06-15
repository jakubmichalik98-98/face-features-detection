import cv2 as cv
from basic_functions import read_image
from luminance_correction import luminance_correction
from mouth_candidates import create_mouth_map_Fg
import matplotlib.pyplot as plt
from face_candidates import fit_ellipse, create_binary_mask

if __name__ == '__main__':
    img = read_image("data/23-1m.bmp")
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    luminance_correction(img_rgb)
    mask = create_binary_mask(img_rgb)
    ellipse_img = fit_ellipse(mask, img_rgb)
    imgYCrCb = cv.cvtColor(ellipse_img, cv.COLOR_RGB2YCrCb)

    mouth_map = create_mouth_map_Fg(imgYCrCb)
    mouth_map_rgb = cv.cvtColor(mouth_map, cv.COLOR_YCrCb2RGB)
    mouth_map_gray = cv.cvtColor(mouth_map_rgb, cv.COLOR_RGB2GRAY)
    ret, th = cv.threshold(mouth_map_gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    # th = cv.bitwise_not(th)
    plt.subplot(1, 2, 1)
    plt.imshow(mouth_map, cmap="gray")
    plt.subplot(1, 2, 2)
    plt.imshow(th, cmap="gray")
    plt.show()
