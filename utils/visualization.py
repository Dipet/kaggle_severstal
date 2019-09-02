import cv2 as cv
import numpy as np


import matplotlib.pyplot as plt


PALET = [(249, 192, 12), (0, 185, 241), (114, 0, 218), (249,50,12)]


def draw_masks(img, masks):
    for j, mask in enumerate(masks):
        mask = mask.astype(np.uint8)

        if np.all(mask == 0):
            continue

        contours, _ = cv.findContours(mask, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
        for i in range(0, len(contours)):
            cv.polylines(img, contours[i], True, PALET[j], 2)


def show_masks(img, masks):
    draw_masks(img, masks)

    for i, color in enumerate(PALET):
        plt.plot(0, 0, '-', c=np.array(color) / 255, label=str(i))

    plt.imshow(img)
    plt.legend()


if __name__ == '__main__':
    from dataset import read_dataset, SteelDataset

    df = read_dataset('../dataset/train.csv', '../dataset/train_images')
    dataset = SteelDataset(df, None, 'valid', catalyst=False)

    for i in dataset:
        show_masks(*i)
        plt.show()
