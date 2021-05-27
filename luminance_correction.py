import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def read_image(img):
    return cv.imread(img)


def show_images(images: list, columns: int, rows: int):
    fig = plt.figure(figsize=(columns, rows))
    for i in range(1, len(images) + 1):
        fig.add_subplot(rows, columns, i)
        plt.imshow(images[i - 1])
        i += 1
    plt.show()


def calculate_luminance(img):
    luminance = np.zeros((480, 640))
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            pixel = img[i, j]
            luminance[i, j] = pixel[0] * 0.2126 + pixel[1] * 0.7152 + pixel[2] * 0.0722
    return luminance


def create_binary_mask(RGB_image):
    imgYCC = cv.cvtColor(RGB_image, cv.COLOR_RGB2YCrCb)
    mask = cv.inRange(imgYCC, (0, 133, 77), (255, 173, 127))
    return mask


if __name__ == '__main__':
    img1 = read_image("D:/Wykrywanie twarzy/face_data/data/07-1m.bmp")
    # mask
    img1_rgb = cv.cvtColor(img1, cv.COLOR_BGR2RGB)
    imgYCC = cv.cvtColor(img1_rgb, cv.COLOR_RGB2YCrCb)
    mask = cv.inRange(imgYCC, (0, 133, 77), (255, 173, 127))
    # th, threshed = cv.threshold(mask, 127, 255, cv.THRESH_BINARY)
    cnts, hiers = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for cnt in cnts:
        if len(cnt) > 50:
            ellipse = cv.fitEllipse(cnt)
            cv.ellipse(img1_rgb, ellipse, (255, 0, 255), 1, cv.LINE_AA)
    plt.subplot(1, 1, 1)
    plt.imshow(img1_rgb, cmap="gray")
    plt.show()
