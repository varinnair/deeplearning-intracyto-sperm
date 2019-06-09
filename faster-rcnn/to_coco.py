import pandas as pd
import os
data = pd.read_csv('annotations-modif.csv')
filenames = data['filename']

# as the images are in train_images folder, add train_images before the image name
for i in range(len(filenames)):
	f = open("coco-files/" + os.path.splitext(filenames[i])[0] + ".txt", "w")
	xmin = data['top_left_x'][i]
	ymin = data['top_left_y'][i]
	xmax = data['bottom_right_x'][i]
	ymax = data['bottom_right_y'][i]
	k = str(filenames[i]) + ',' + str(xmin) + ',' + str(ymin) + ',' + str(xmax - xmin) + ',' + str(ymax - ymin) + ',' + data['normal_abnormal'][i]
	f.write(str(k))
	f.close()