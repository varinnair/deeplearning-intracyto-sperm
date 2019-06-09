from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from SpermDataset import SpermDataset, ToTensor
import warnings
import json
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode
def show_sperms(image, sperms):
    """Show image with landmarks"""
    plt.imshow(image)
    plt.scatter(sperms[:, 0], sperms[:, 1], s=10, marker='.', c='r')
    plt.pause(0.001)  # pause a bit so that plots are updated

def sanity_check():
	sperm_dataset = SpermDataset(csv_file='annotations2-modif.csv',
	                                    root_dir='images/')
	fig = plt.figure()
	for i in range(len(sperm_dataset)):
		sample = sperm_dataset[i]
		print(i, sample['image'].shape, sample['sperms'].shape)
		ax = plt.subplot(1, 4, i + 1)
		plt.tight_layout()
		ax.set_title('Sample #{}'.format(i))
		ax.axis('off')
		show_sperms(**sample)
		if i == 3:
			plt.show()
			break

def process_csv(input, output):
	sperm_data = pd.read_csv(input)
	bbox = json.loads(sperm_data.iloc[0, 2])
	for index in range(2, 6):
		sperm_data.insert(index, list(bbox.keys())[index - 1], -index)
	for index, bbox in enumerate(sperm_data.iloc[:, 6]):
		curr = json.loads(bbox)
		del curr['name']
		for key in curr.keys():
			sperm_data.loc[index, key] = curr[key]
	sperm_data = sperm_data.drop(columns=['bbox_coords', 'flip_type'])
	sperm_data.to_csv(output, index=None, header=True)

# Helper function to show a batch
def show_sperms_batch(sample_batched):
	images_batch, sperms_batch = sample_batched['image'], sample_batched['sperms']
	batch_size = len(images_batch)
	im_size = images_batch.size(2)
	grid_border_size = 6
	grid = utils.make_grid(images_batch)
	plt.imshow(grid.numpy().transpose((1, 2, 0)))
	for i in range(batch_size):
		plt.scatter(sperms_batch[i, :, 0].numpy() + i * im_size + (i + 1) * grid_border_size,
                    sperms_batch[i, :, 1].numpy() + grid_border_size,
                    s=10, marker='.', c='r')
		plt.title('Batch from dataloader')


transformed_dataset = SpermDataset(csv_file='annotations-modif.csv',
                                           root_dir='images/',
                                           transform=transforms.Compose([
                                               ToTensor()
                                           ]))
dataloader = DataLoader(transformed_dataset, batch_size=8,
                        shuffle=True, num_workers=4)
for i_batch, sample_batched in enumerate(dataloader):
	print(i_batch, sample_batched['image'].size(),
          sample_batched['sperms'].size())
	if i_batch == 6:
		plt.figure()
		show_sperms_batch(sample_batched)
		plt.axis('off')
		plt.ioff()
		plt.show()