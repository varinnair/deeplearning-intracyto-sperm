from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

plt.ion()   # interactive mode
X, y = data, y = data.iloc[:, 1:5]
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2)



from pascal_voc_writer import Writer
import pandas as pd
sperm_data = pd.read_csv('faster-rcnn-data/annotations-modif.csv')
img_names = sperm_data.iloc[:, 0]
for img_name in img_names:
	writer = Writer('faster-rcnn-data/images/' + img_name, 776, 582)
	select_data = sperm_data.loc[sperm_data['filename']==img_name]
	for index, row in select_data.iterrows():
		writer.addObject(row['normal_abnormal'], row['top_left_x'], row['top_left_y'], row['bottom_right_x'], row['bottom_right_y'])
	writer.save('faster-rcnn-data/annotations/' + os.path.splitext(img_name)[0]+'.xml')