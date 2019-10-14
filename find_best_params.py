import torch
from utils.dataset import read_dataset, BinaryClassSteelDataset, TrainSteelDataset, MultiClassSteelDataset
import albumentations as A
from torch.utils.data import DataLoader


from mlcomp.contrib.transform.albumentations import ChannelTranspose
from mlcomp.contrib.dataset.classify import ImageDataset
from mlcomp.contrib.transform.rle import rle2mask, mask2rle
from mlcomp.contrib.transform.tta import TtaWrap

import numpy as np

import os
from tqdm import tqdm

df = read_dataset('/mnt/HDD/home/druzhinin/kaggle/kaggle_severstal/dataset/train.csv',
                  '/mnt/HDD/home/druzhinin/kaggle/kaggle_severstal/dataset/train_images')

def create_transforms(additional):
    res = list(additional)
    # add necessary transformations
    res.extend([
        A.Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        ),
        ChannelTranspose()
    ])
    res = A.Compose(res)
    return res

# Different transforms for TTA wrapper
transforms = [
    [],
    [A.HorizontalFlip(p=1)],
    [A.VerticalFlip(p=1)],
    [A.HorizontalFlip(p=1), A.VerticalFlip(p=1)]
]

transforms = [create_transforms(t) for t in transforms]

# datasets = [TtaWrap(BinaryClassSteelDataset(df, transforms=t), tfms=t) for t in transforms]
# loaders = [DataLoader(d, num_workers=0, batch_size=16, shuffle=False) for d in datasets]

device = 'cuda'

mobilenet_cls = torch.jit.load('/mnt/HDD/home/druzhinin/kaggle/kaggle_severstal/download/12mobilenet-severstal/torchscript.pth',
                               map_location=device)
resnet34_cls = torch.jit.load('/mnt/HDD/home/druzhinin/kaggle/kaggle_severstal/download/resnet34-class01/torchscript.pth',
                              map_location=device)


def predict_tta_class(models: dict, loaders_batch):
    results = {i: [] for i in models}
    data = None

    for i, data in enumerate(loaders_batch):
        features = data.pop('features').cuda()
        for key, model in models.items():
            results[key].append(torch.sigmoid(model(features)))
        data = data

    # TTA mean
    for key in list(results.keys()):
        preds = torch.stack(results.pop(key))
        preds = torch.mean(preds, dim=0).squeeze()
        results[key] = preds.detach().cpu().numpy()

    return results, data


def find_best_multiclass(df, tta,  model, thresholds):
    datasets = [TtaWrap(MultiClassSteelDataset(df, transforms=t), tfms=t) for t in tta]
    loaders = [DataLoader(d, num_workers=6, batch_size=16, shuffle=False) for d in datasets]

    results = {i: [] for i in thresholds}

    for batch in tqdm(zip(*loaders), total=len(loaders[0])):
        preds, data = predict_tta_class({'model': model}, batch)
        preds = preds['model']
        true = data['targets'].detach().cpu().numpy()

        for t in thresholds:
            p = (preds > t).astype(int)
            results[t].append((p == true).astype(int))

    shape = results[list(results.keys())[0]][0].shape
    accuracy = {cls: {} for cls in range(shape[1])}
    for t, values in results.items():
        values = np.concatenate(values, axis=0)

        for cls in list(accuracy.keys()):
            accuracy[cls][t] = np.sum(values[:, cls]) / len(values[:, cls])

    bests = {}
    for cls, (key, data) in enumerate(accuracy.items()):
        d = [(i, val) for i, val in data.items()]
        d = sorted(d, key=lambda x: x[1])
        print(f'Class: {cls}; Best: {d[-1]} Worse: {d[0]} Mean: {np.mean([i for _, i in d])}')
        bests[cls] = d[-1]

    return bests


find_best_multiclass(df, transforms, resnet34_cls, np.arange(0.05, 1.0, 0.05))

# res = []
# without_mask = set()
# # Iterate over all TTA loaders
# total = len(loaders[0])
# for loaders_batch in tqdm(zip(*loaders), total=total):
#     preds, data = predict_tta({'mobile': mobilenet_cls,
#                                'resnet': resnet34_cls},
#                               loaders_batch)
#
#     # Batch post processing
#     for file, p_bin, p_cls in zip(data['img_paths'], preds['mobile'], preds['resnet']):
#         file = os.path.basename(file)
#
#         _p = p_cls > 0.5
#         p_bin = (p_bin + np.max(_p)) > 0.55
#         # Image postprocessing
#         for i in range(4):
#             imageid_classid = file + '_' + str(i + 1)
#             if not p_bin or not _p[i]:
#                 res.append({
#                     'ImageId_ClassId': imageid_classid,
#                     'EncodedPixels': ''})