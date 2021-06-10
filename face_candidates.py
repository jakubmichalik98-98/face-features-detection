import cv2 as cv
import matplotlib.pyplot as plt
from copy import deepcopy

def read_image(img):
    return cv.imread(img)


def create_binary_mask(rgb_image):
    imgYCrCb = cv.cvtColor(rgb_image, cv.COLOR_RGB2YCrCb)
    mask = cv.inRange(imgYCrCb, (0, 133, 77), (255, 173, 127))
    return mask


def fit_ellipse(mask_img, rgb_image):
    contours, hierarchy = cv.findContours(mask_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    elipse_mask = deepcopy(rgb_image)
    h = rgb_image.shape[0]
    w = rgb_image.shape[1]
    for contour in contours:
        if len(contour) > 50:
            ellipse = cv.fitEllipse(contour)
            elipse_mask = cv.ellipse(elipse_mask, ellipse, (255, 255, 255), -1, cv.LINE_AA)
            for i in range(h):
                for j in range(w):
                    if((elipse_mask[i,j] != [255, 255, 255]).all()):
                        elipse_mask[i,j] = [0, 0, 0]

    for i in range(h):
        for j in range(w):
            if((elipse_mask[i, j] != [255, 255, 255]).all()):
                rgb_image[i, j] = [0, 0, 0]    
            
    return rgb_image


if __name__ == '__main__':
    """NA RAZIE TAK ZOSTAWIAM ZEBYS WIEDZIAŁ JAK UZYWAC FUNKCJI"""
    img1 = read_image("D:/Wykrywanie twarzy/face_data/data/11-1m.bmp")
    img1_rgb = cv.cvtColor(img1, cv.COLOR_BGR2RGB)
    mask = create_binary_mask(img1_rgb)
    img_to_show = fit_ellipse(mask, img1_rgb)
    plt.subplot(1, 1, 1)
    plt.imshow(img_to_show)
    plt.show()
