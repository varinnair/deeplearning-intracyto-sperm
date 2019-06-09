import csv
import json
import cv2
import os
import numpy as np
# ['#filename', 'file_size', 'file_attributes', 'region_count', 'region_id', 'region_shape_attributes', 'region_attributes']
#       0           1               2               3               4                   5                       6
cwd = os.getcwd()

read_file = 'region_data_1.csv' # change to region_data_2.csv for second run
write_file = 'new1.csv' # change to new2.csv for second run

with open(read_file) as csvfile:
    readfile = csv.reader(csvfile, delimiter=',')
    with open(write_file, 'w', newline='') as writefile:
        filewriter = csv.writer(writefile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        i = 0
        for row in readfile:
            if i == 0:
                filewriter.writerow(['filename', 'region_shape_attr', 'region_attr'])
                i += 1
                continue

            filename = row[0]

            shape_attributes = row[5]
            shape_attributes = json.loads(shape_attributes)
            if shape_attributes['name'] == 'rect':
                top_left_x = shape_attributes['x']
                top_left_y = shape_attributes['y']
                bottom_right_x = top_left_x + shape_attributes['width']
                bottom_right_y = top_left_y + shape_attributes['height']

            elif shape_attributes['name'] == 'polygon':
                # get rect coords from manav's function
                # scale in same way
                all_x = shape_attributes['all_points_x']
                all_y = shape_attributes['all_points_y']
                
                pts = []
                for j in range(len(all_x)):
                    x_j = all_x[j]
                    y_j = all_y[j]
                    pts.append([x_j, y_j])
                pts = np.array(pts)

                x, y, w, h = cv2.boundingRect(pts)

                top_left_x = x
                top_left_y = y
                bottom_right_x = x + w
                bottom_right_y = y + h

            new_shape_attributes = {}
            new_shape_attributes['name'] = 'rect'
            new_shape_attributes['top_left_x'] = top_left_x
            new_shape_attributes['top_left_y'] = top_left_y
            new_shape_attributes['bottom_right_x'] = bottom_right_x
            new_shape_attributes['bottom_right_y'] = bottom_right_y
            new_shape_attributes_str = json.dumps(new_shape_attributes)

            region_attributes = row[6]
            filename_pre, extension = filename.split('.')
            filewriter.writerow([filename_pre+'_resized.'+extension, new_shape_attributes_str, region_attributes])
            i += 1