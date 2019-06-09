import csv
import json
import cv2
import os
import scipy.misc
from PIL import Image
import numpy as np

read_file_name = ''
write_file_name = ''

with open(read_file_name) as csvfile:
    readfile = csv.reader(csvfile, delimiter=',')
    with open(write_file_name, 'w', newline='') as writefile:
        filewriter = csv.writer(writefile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        i = 0
        for row in readfile:
            if i == 0:
                filewriter.writerow(['filename', 'label'])
                i += 1
                continue
            
            filename, normal_abnormal = row
            normal_abnormal_json = json.loads(normal_abnormal)
            filewriter.writerow([filename, normal_abnormal_json['Normal/Abnormal']])
            i += 1