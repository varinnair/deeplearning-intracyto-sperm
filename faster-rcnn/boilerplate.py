import pandas as pd
test = pd.read_csv('test.csv')
data = pd.DataFrame()
data['format'] = test['filename']

# as the images are in train_images folder, add train_images before the image name
for i in range(data.shape[0]):
    data['format'][i] = 'train_images/' + data['format'][i]

# add xmin, ymin, xmax, ymax and class as per the format required
for i in range(data.shape[0]):
    data['format'][i] = data['format'][i] + ',' + str(test['top_left_x'][i]) + ',' + str(test['top_left_y'][i]) + ',' + str(test['bottom_right_x'][i]) + ',' + str(test['bottom_right_y'][i]) + ',' + test['normal_abnormal'][i]

data.to_csv('annotate-test.txt', header=None, index=None, sep=' ')