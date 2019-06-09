import pandas as pd
import os
from shutil import copyfile
data = pd.read_csv('validation.csv')
filenames = data['filename']

# as the images are in train_images folder, add train_images before the image name
for i in range(len(filenames)):
	copyfile('images/' + filenames[i], 'validation_images/' + filenames[i])