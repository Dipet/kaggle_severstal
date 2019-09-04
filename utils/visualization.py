import cv2 as cv
import numpy as np


import matplotlib.pyplot as plt


PALET = [(249, 192, 12), (0, 185, 241), (114, 0, 218), (249,50,12)]


def draw_masks(img, masks, orig_masks=None):
    if orig_masks is None:
        orig_masks = [None] * len(masks)

    for j, (mask, omask) in enumerate(zip(masks, orig_masks)):
        mask = mask.astype(np.uint8)

        if omask is not None:
            omask = omask.astype(np.bool)
            color = np.clip(np.array(PALET[j]) * 2, 0, 255)
            img[omask] = color

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
    from dataset import read_dataset, SteelDataset, get_train_transforms

    df = read_dataset('../dataset/train.csv', '../dataset/train_images')
    transforms = get_train_transforms(mean=(0.485, 0.456, 0.406),
                                      std=(0.229, 0.224, 0.225),)
    dataset = SteelDataset(df, transforms, 'valid', catalyst=False)

    for img, mask in dataset:
        img = img.cpu().numpy()
        mask = mask.cpu().numpy()

        img = np.transpose(img, [1, 2, 0])
        show_masks(img, mask)
        plt.show()
