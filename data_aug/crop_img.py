import csv
import json
import cv2
import os
import numpy as np

csv_filename = ''

cwd = os.getcwd()

read_file_name = 'new1.csv' # change to new2.csv for second run
write_file_name = 'cropped1.csv' # change to cropped2.csv for second run
write_folder_name = 'cropped_data_1' # change to cropped_data_2 for second run
read_folder_name = 'resized_data_1' # change  to resized_data_2 for second run

with open(read_file_name) as csvfile:
    readfile = csv.reader(csvfile, delimiter=',')
    with open(write_file_name, 'w', newline='') as writefile:
        filewriter = csv.writer(writefile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        i = 0
        for row in readfile:
            if i == 0:
                filewriter.writerow(['filename', 'normal_abnormal'])
                i += 1
                continue
            
            filename, bbox, normal_abnormal = row
            image = cv2.imread(cwd+'/'+read_folder_name+'/'+filename)

            bbox_json = json.loads(bbox)
            top_left_x = bbox_json['top_left_x']
            top_left_y = bbox_json['top_left_y']
            bottom_right_x = bbox_json['bottom_right_x']
            bottom_right_y = bbox_json['bottom_right_y']

            cropped_img = image[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
            cv2.imwrite(cwd + '/' + write_folder_name + '/cropped_' + str(i) + '_'+ filename, cropped_img)

            filewriter.writerow(['cropped_' + str(i) + '_'+ filename, normal_abnormal])
            i += 1
