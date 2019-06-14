import csv
import json
import cv2
import os
import numpy as np

csv_filename = ''

cwd = os.getcwd()

# create two directories - 'normal_sperm/' and 'abnormal_sperm/'
read_file_name = 'annotations-modif.csv' # change to new2.csv for second run
read_folder_name = 'faster-rcnn-data/images' # change  to resized_data_2 for second run

with open(read_file_name) as csvfile:
    readfile = csv.reader(csvfile, delimiter=',')
    i = 0
    for row in readfile:
        if i == 0:
            i += 1
            continue
            # top_left_x  top_left_y  bottom_right_x  bottom_right_y
        # filename, bbox, normal_abnormal = row

        filename, top_left_x, top_left_y, bottom_right_x, bottom_right_y, normal_abnormal = row
        image = cv2.imread(cwd+'/'+read_folder_name+'/'+filename)

        # bbox_json = json.loads(bbox)
        # top_left_x = bbox_json['top_left_x']
        # top_left_y = bbox_json['top_left_y']
        # bottom_right_x = bbox_json['bottom_right_x']
        # bottom_right_y = bbox_json['bottom_right_y']

        top_left_x = int(top_left_x)
        top_left_y = int(top_left_y)
        bottom_right_x = int(bottom_right_x)
        bottom_right_y = int(bottom_right_y)

        normal_abnormal_json = json.loads(normal_abnormal)
        cropped_img = image[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
        
        if normal_abnormal_json['Normal/Abnormal'] == 'N':
            write_folder_name = 'new_classification/normal_sperm'
        else:
            write_folder_name = 'new_classification/abnormal_sperm'

        cropped_img = image[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
        cv2.imwrite(cwd + '/' + write_folder_name + '/cropped_' + str(i) + '_'+ filename, cropped_img)
        i += 1
