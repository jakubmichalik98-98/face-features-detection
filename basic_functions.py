import cv2 as cv
import matplotlib.pyplot as plt


def read_image(img):
    """
    Function responsible for read an image
    :param img: Path to image
    :return: Image read from the path
    """
    return cv.imread(img)


def show_images(images: list, columns: int, rows: int):
    """
    Function responsible for show images
    :param images: How much images want to display
    :param columns: How much columns in subplot
    :param rows: How much rows in subplot
    :return: Subplot with images
    """
    fig = plt.figure(figsize=(4 * columns, 4 * rows))
    for i in range(1, len(images) + 1):
        fig.add_subplot(rows, columns, i)
        plt.imshow(images[i - 1])
        i += 1
    plt.show()
