from sklearn.manifold import TSNE
import numpy as np
import os
import cv2
import csv
from PIL import Image
from sklearn.decomposition import PCA

cwd = os.getcwd()
read_file_name = 'train.csv'
feature_vectors = []
with open(read_file_name) as csvfile:
    readfile = csv.reader(csvfile, delimiter=',')
    i = 0
    for row in readfile:
        if i == 0:
            i += 1
            continue
        filename, label = row
        img = np.array(Image.open(cwd + '/data/train/' + filename).resize((25, 25)))
        img = img.flatten()
        feature_vectors.append(img)
        i += 1

print("done with feature vectors")

model = TSNE(n_components=2, random_state=0)
points = model.fit_transform(feature_vectors) # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE


import matplotlib.pyplot as plt
plt.scatter(points[:,0], points[:, 1])
plt.show()

cwd = os.getcwd()
read_file_name = 'train.csv'
feature_vectors = []
target_v = []
with open(read_file_name) as csvfile:
    readfile = csv.reader(csvfile, delimiter=',')
    i = 0
    for row in readfile:
        if i == 0:
            i += 1
            continue
        filename, label = row
        if label == 'N':
            target_v.append(0)
        else:
            target_v.append(1)
        i += 1
