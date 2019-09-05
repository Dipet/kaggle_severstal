from torch.utils.data import Dataset, DataLoader

import os
import cv2 as cv
import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split


from albumentations import Compose, Normalize, Flip, HorizontalFlip
from albumentations.pytorch import ToTensor



def mask2rle(img):
    '''https://www.kaggle.com/paulorzp/rle-functions-run-lenght-encode-decode

    Args:
        img: numpy array, 1 - mask, 0 - background

    Returns: run length as string formated
    '''
    pixels= img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def rle_to_mask(rle):
    label = rle.split(" ")
    positions = map(int, label[0::2])
    length = map(int, label[1::2])
    mask = np.zeros(256 * 1600, dtype=np.uint8)
    for pos, le in zip(positions, length):
        mask[pos:(pos + le)] = 1
    return mask.reshape(256, 1600, order='F')


class SteelDataset(Dataset):
    def __init__(self, df, transforms, phase='train', catalyst=True, binary=False):
        self.df = df
        self.transforms = transforms
        self.phase = phase
        self.catalyst = catalyst
        self.binary = binary

    def make_mask(self, row_id):
        '''Given a row index, return image_id and mask (256, 1600, 4)'''
        labels = self.df.iloc[row_id][:4]
        masks = np.zeros((256, 1600, 4), dtype=np.float32)  # float32 is V.Imp
        # 4:class 1～4 (ch:0～3)

        for idx, label in enumerate(labels.values):
            if label is not np.nan:
                masks[:, :, idx] = rle_to_mask(label)
        return masks

    def _get_train_valid(self, idx):
        img_path = self.df.iloc[idx].name

        img = cv.imread(img_path, cv.IMREAD_COLOR)
        mask = self.make_mask(idx)

        if self.transforms:
            augmented = self.transforms(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']  # 1x256x1600x4

        if isinstance(mask, np.ndarray):
            mask = np.transpose(mask, [2, 0, 1])
        else:
            mask = mask[0].permute(2, 0, 1)  # 1x4x256x1600

        if self.catalyst:
            return {'targets': mask, 'features': img}
        else:
            return img, mask

    def _get_infer(self, idx):
        img_path = self.df.iloc[idx].name

        img = cv.imread(img_path, cv.IMREAD_COLOR)

        if self.transforms:
            augmented = self.transforms(image=img)
            img = augmented['image']

        if self.catalyst:
            return {'features': img}
        else:
            return img

    def _get_binary(self, idx):
        img_path = self.df.iloc[idx].name

        img = cv.imread(img_path, cv.IMREAD_COLOR)
        mask = self.make_mask(idx)

        if self.transforms:
            augmented = self.transforms(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']  # 1x256x1600x4

        if isinstance(mask, np.ndarray):
            cls = np.any(mask != 0).astype(int)
        else:
            cls = torch.Tensor([torch.any(mask != 0).long()])

        if self.catalyst:
            return {'targets': cls, 'features': img}
        else:
            return img, cls

    def __getitem__(self, idx):
        if self.phase == 'infer':
            return self._get_infer(idx)

        if self.binary:
            return self._get_binary(idx)

        if self.phase in ['train', 'valid', 'test']:
            return self._get_train_valid(idx)

        raise Exception('Something wrong')

    def __len__(self):
        return len(self.df)


def get_train_transforms(mean, std):
    transforms = [Flip(),
                  Normalize(mean=mean, std=std),
                  ToTensor()]

    return Compose(transforms)


def get_inference_transforms(mean, std):
    transforms = [Normalize(mean, std),
                  ToTensor()]

    return Compose(transforms)


def read_dataset(path, data_folder):
    df = pd.read_csv(path)

    # some preprocessing
    # https://www.kaggle.com/amanooo/defect-detection-starter-u-net
    df['ImageId'], df['ClassId'] = zip(*df['ImageId_ClassId'].str.split('_'))
    df['ImageId'] = df['ImageId'].apply(lambda x: os.path.join(data_folder, x))
    df['ClassId'] = df['ClassId'].astype(int)
    df = df.pivot(index='ImageId', columns='ClassId', values='EncodedPixels')
    df['defects'] = df.count(axis=1)

    return df


def get_dataloader(df, transforms, batch_size, shuffle, num_workers,
                   phase, catalyst, pin_memory, binary):
    dataset = SteelDataset(df, transforms, phase, catalyst, binary=binary)
    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=num_workers,
                      pin_memory=pin_memory)


def get_train_val_datasets(df, data_folder=None, mean=None, std=None,
                           catalyst=True, binary=False, full_train=False):
    if isinstance(df, str):
        df = read_dataset(df, data_folder)

    train_df, val_df = train_test_split(df, test_size=0.2,
                                        stratify=df["defects"])

    train_transforms = get_train_transforms(mean, std)
    val_transforms = get_inference_transforms(mean, std)

    if full_train:
        train_dataset = SteelDataset(df, train_transforms, 'train', catalyst, binary=binary)
    else:
        train_dataset = SteelDataset(train_df, train_transforms, 'train', catalyst, binary=binary)
    val_dataset = SteelDataset(val_df, val_transforms, 'valid', catalyst, binary=binary)

    return train_dataset, val_dataset


def get_train_val_dataloaders(
        df,
        data_folder=None,
        mean=None,
        std=None,
        batch_size=8,
        num_workers=4,
        catalyst=False,
        pin_memory=False,
        binary=False,
        full_train=False,
):
    train_dataset, val_dataset = get_train_val_datasets(df, data_folder, mean, std, catalyst, binary=binary, full_train=full_train)

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=num_workers,
                                pin_memory=pin_memory)

    return train_dataloader, val_dataloader
