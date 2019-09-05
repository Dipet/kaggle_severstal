import numpy as np
import pandas as pd


def rle_area(rle):
    label = rle.split(" ")
    length = list(map(int, label[1::2]))
    return np.sum(length)


df = pd.read_csv('dataset/train.csv')
print(df.head())

areas = []
c = 0
val = 0
for rle in df['EncodedPixels'].values:
    if rle is np.nan:
        continue

    c += 1
    val += rle_area(rle)
    if c >= 0:
        areas.append(val)
        val = 0
        c = 0
areas = np.array(areas)

print(np.min(areas), np.max(areas), np.mean(areas))
print(np.quantile(areas, [0.05, 0.1, 0.25, 0.5, 0.75]))
