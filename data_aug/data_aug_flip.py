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

def get_lr_flipped_coords(bbox_top_left_x, bbox_top_left_y, bbox_w, bbox_h):
    bbox_top_right_x = bbox_top_left_x + bbox_w
    bbox_top_right_y = bbox_top_left_y

    flipped_top_left_x = (IMG_WIDTH - 1) - bbox_top_right_x
    flipped_top_left_y = bbox_top_right_y

    return flipped_top_left_x, flipped_top_left_y

def get_ud_flipped_coords(bbox_top_left_x, bbox_top_left_y, bbox_w, bbox_h):
    bbox_bottom_left_x = bbox_top_left_x
    bbox_bottom_left_y = bbox_top_left_y + bbox_h

    flipped_top_left_x = bbox_bottom_left_x
    flipped_top_left_y = (IMG_HEIGHT - 1) - bbox_bottom_left_y
    
    return flipped_top_left_x, flipped_top_left_y

visited_imgs = set()

read_file_name = 'new1.csv' # change to 'new2.csv' for second run
write_file_name = 'flipped1.csv' # change to 'flipped2.csv' for second run

read_img_dir = 'resized_data_1' # change to 'resized_data_2' for second run
write_img_dir = 'flipped_data_1' # change to 'flipped_data_2' for second run

with open(read_file_name) as csvfile:
    readfile = csv.reader(csvfile, delimiter=',')
    with open(write_file_name, 'w', newline='') as writefile:
        filewriter = csv.writer(writefile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        i = 0
        for row in readfile:
            if i == 0:
                filewriter.writerow(['filename', 'flip_type', 'bbox_coords', 'normal_abnormal'])
                i += 1
                continue

            filename, bbox, normal_abnormal = row
            if filename not in visited_imgs:
                img = Image.open(cwd + '/' + read_img_dir + '/' + filename)
                img = np.array(img)

                # flipping img
                flipped_img_lr = np.fliplr(img)
                flipped_img_ud = np.flipud(img)
                flipped_img_lr_ud = np.flipud(flipped_img_lr)

                # creating new file names for flipped imgs
                filename_pre, extension = filename.split('.')
                new_file_name_1 = filename_pre + '_' + 'lr' + '.' + extension
                new_file_name_2 = filename_pre + '_' + 'ud' + '.' + extension
                new_file_name_3 = filename_pre + '_' + 'lr_ud' + '.' + extension

                # saving new flipped imgs
                scipy.misc.imsave(cwd + "/" + write_img_dir + "/" + new_file_name_1, flipped_img_lr)
                scipy.misc.imsave(cwd + "/" + write_img_dir + "/" + new_file_name_2, flipped_img_ud)
                scipy.misc.imsave(cwd + "/" + write_img_dir + "/" + new_file_name_3, flipped_img_lr_ud)
                
            bbox_json = json.loads(bbox)
            top_left_x = bbox_json['top_left_x']
            top_left_y = bbox_json['top_left_y']
            bottom_right_x = bbox_json['bottom_right_x']
            bottom_right_y = bbox_json['bottom_right_y']
            
            bbox_width = bottom_right_x - top_left_x
            bbox_height = bottom_right_y - top_left_y

            # lr flip (horizontal flip)
            flipped_top_left_x, flipped_top_left_y = get_lr_flipped_coords(top_left_x, top_left_y, bbox_width, bbox_height)
            new_shape_attributes = {}
            new_shape_attributes['name'] = 'rect'
            new_shape_attributes['top_left_x'] = flipped_top_left_x
            new_shape_attributes['top_left_y'] = flipped_top_left_y
            new_shape_attributes['bottom_right_x'] = flipped_top_left_x + bbox_width
            new_shape_attributes['bottom_right_y'] = flipped_top_left_y + bbox_height
            new_shape_attributes_str = json.dumps(new_shape_attributes)
            filewriter.writerow([filename, 'lr', new_shape_attributes_str, normal_abnormal])
            
            # ud flip (vertical flip)
            flipped_top_left_x, flipped_top_left_y = get_ud_flipped_coords(top_left_x, top_left_y, bbox_width, bbox_height)
            new_shape_attributes = {}
            new_shape_attributes['name'] = 'rect'
            new_shape_attributes['top_left_x'] = flipped_top_left_x
            new_shape_attributes['top_left_y'] = flipped_top_left_y
            new_shape_attributes['bottom_right_x'] = flipped_top_left_x + bbox_width
            new_shape_attributes['bottom_right_y'] = flipped_top_left_y + bbox_height
            new_shape_attributes_str = json.dumps(new_shape_attributes)
            filewriter.writerow([filename, 'ud', new_shape_attributes_str, normal_abnormal])
            
            # lr + ud flip (horizontal + vertical flips)
            flipped_top_left_x, flipped_top_left_y = get_lr_flipped_coords(top_left_x, top_left_y, bbox_width, bbox_height)
            flipped_top_left_x, flipped_top_left_y = get_ud_flipped_coords(flipped_top_left_x, flipped_top_left_y, bbox_width, bbox_height)
            new_shape_attributes = {}
            new_shape_attributes['name'] = 'rect'
            new_shape_attributes['top_left_x'] = flipped_top_left_x
            new_shape_attributes['top_left_y'] = flipped_top_left_y
            new_shape_attributes['bottom_right_x'] = flipped_top_left_x + bbox_width
            new_shape_attributes['bottom_right_y'] = flipped_top_left_y + bbox_height
            new_shape_attributes_str = json.dumps(new_shape_attributes)
            filewriter.writerow([filename, 'lr-ud', new_shape_attributes_str, normal_abnormal])
            
            visited_imgs.add(filename)
            i += 1