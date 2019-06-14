import csv
import json
import cv2
import os
import scipy.misc
from PIL import Image
import numpy as np

IMG_WIDTH = 776
IMG_HEIGHT = 582

cwd = os.getcwd()

visited_imgs = set()

read_file_name = "annotations-modif.csv" # change to 'new2.csv' for second run

write_img_dir = "labels/"

with open(read_file_name) as csvfile:
    readfile = csv.reader(csvfile, delimiter=',')
    next(readfile)
    for row in readfile:
        write_file_name = write_img_dir + row[0].split('.')[0] + ".txt"
        x1 = int(row[1])
        y1 = int(row[2])
        x2 = int(row[3])
        y2 = int(row[4])
        center_x = 1/IMG_WIDTH * (x1 + x2)/2
        center_y = 1/IMG_HEIGHT * (y1 + y2)/2
        width = 1/IMG_WIDTH * (x2 - x1)
        height = 1/IMG_HEIGHT * (y2 - y1)

        abnormal_str = json.loads(row[5])["Normal/Abnormal"] #1
        if abnormal_str == "A":
            abnormal = "1"
        else:
            abnormal = "0"
        abnormal = "0"

        with open(write_file_name, 'a+', newline='') as writefile:
            writefile.write(abnormal + ' ' + str(center_x) + ' ' + str(center_y) + ' ' + str(width) + ' ' + str(height) + '\n')
            
print("done")