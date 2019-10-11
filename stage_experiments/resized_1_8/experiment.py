from collections import OrderedDict

from catalyst.dl import ConfigExperiment

from utils.dataset import *

from sklearn.model_selection import train_test_split
from torch.nn import BCEWithLogitsLoss


class Experiment(ConfigExperiment):
    @staticmethod
    def get_transforms(stage: str = None, mode: str = None):
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

        if mode == 'valid':
            return get_inference_transforms(mean=mean, std=std)

        transforms = []
        if stage == 'stage1':
            transforms += [albu.RandomCrop(256, 256, p=1)]
        elif stage == 'stage2':
            transforms += [albu.RandomCrop(256, 1024, p=1)]
        else:
            transforms += [albu.OneOf([
                ShiftScaleRotate(shift_limit=0.25,
                                 scale_limit=0.25,
                                 rotate_limit=90,
                                 border_mode=cv.BORDER_CONSTANT,
                                 value=0,
                                 mask_value=0,
                                 p=0.5),
                albu.RandomSizedCrop(min_max_height=(200, 256), height=256,
                                     width=1600, w2h_ratio=1600 / 256, p=0.5)
            ])]

        transforms += [
            albu.RandomBrightnessContrast(),
            albu.RandomGamma(),
            Flip(),
            Normalize(mean=mean, std=std),
            ToTensor()
        ]

        return albu.Compose(transforms)

    def get_datasets(self, stage: str, **kwargs):
        df = read_dataset('/mnt/HDD/home/druzhinin/kaggle/kaggle_severstal/dataset/train.csv',
                          '/mnt/HDD/home/druzhinin/kaggle/kaggle_severstal/dataset/train_images')

        train, test = train_test_split(df, test_size=0.05,
                                       stratify=df["defects"])

        trainset = SteelDataset(train, self.get_transforms(stage, mode='train'),
                                phase='train', catalyst=True, binary=False, multi=False)
        testset = SteelDataset(test, self.get_transforms(stage, mode='valid'),
                               phase='valid', catalyst=True, binary=False, multi=False)

        datasets = OrderedDict()
        datasets["train"] = trainset
        datasets["valid"] = testset

        return datasets
