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


if __name__ == '__main__':
    img1 = read_image("data/01-1m.bmp")
    img2 = read_image("data/05-1m.bmp")
    images = [img1, img2]
    show_images(images, 2, 1)

