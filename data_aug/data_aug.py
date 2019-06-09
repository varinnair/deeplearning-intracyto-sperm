import csv
import json
import cv2
import os

cwd = os.getcwd()

def padding(image, top_pad=0, bottom_pad=0, left_pad=0, right_pad=0):
    WHITE = [255, 255, 255]
    constant = cv2.copyMakeBorder(image,top_pad,bottom_pad,left_pad,right_pad,cv2.BORDER_CONSTANT,value=WHITE)
    return constant

visited_imgs = set()

read_file_name = 'new1.csv' # change to 'new2.csv' for second run
write_file_name = 'padded1.csv' # change to 'padded2.csv' for second run

read_img_dir = 'resized_data_1' # change to 'resized_data_2' for second run
write_img_dir = 'padded_data_1' # change to 'padded_data_2' for second run

step = 1 # can change this to 2, or 4 depending on how much data you want to get, and how fast you want it

with open(read_file_name) as csvfile:
    readfile = csv.reader(csvfile, delimiter=',')
    with open(write_file_name, 'w', newline='') as writefile:
        filewriter = csv.writer(writefile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        i = 0
        for row in readfile:
            if i == 0:
                filewriter.writerow(['filename', 'left_pad', 'top_pad', 'bbox_coords', 'normal_abnormal'])
                i += 1
                continue
            filename, bbox, normal_abnormal = row
            for width_left_pad in range(0, 25, step):
                width_right_pad = 24 - width_left_pad
                for height_top_pad in range(0, 19, step):
                    height_bottom_pad = 18 - height_top_pad

                    if filename not in visited_imgs:
                        img = cv2.imread(cwd + '/' + read_img_dir + '/' + filename)
                        padded_img = padding(img, height_top_pad, height_bottom_pad, width_left_pad, width_right_pad)
                        filename_pre, extension = filename.split('.')
                        new_file_name = filename_pre + '_' + str(width_left_pad) + '_' + str(height_top_pad) + '.' + extension
                        cv2.imwrite(cwd + "/" + write_img_dir + "/" + new_file_name, padded_img)

                    bbox_json = json.loads(bbox)
                    top_left_x = bbox_json['top_left_x']
                    top_left_y = bbox_json['top_left_y']
                    bottom_right_x = bbox_json['bottom_right_x']
                    bottom_right_y = bbox_json['bottom_right_y']

                    new_shape_attributes = {}
                    new_shape_attributes['name'] = 'rect'
                    new_shape_attributes['top_left_x'] = top_left_x + width_left_pad
                    new_shape_attributes['top_left_y'] = top_left_y + height_top_pad
                    new_shape_attributes['bottom_right_x'] = bottom_right_x + width_left_pad
                    new_shape_attributes['bottom_right_y'] = bottom_right_y + height_top_pad
                    new_shape_attributes_str = json.dumps(new_shape_attributes)

                    filewriter.writerow([filename, width_left_pad, height_top_pad, new_shape_attributes_str, normal_abnormal])
            visited_imgs.add(filename)
            i += 1