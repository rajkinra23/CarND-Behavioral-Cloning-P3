'''
This library contains methods to open and structure the training data
for our model.
'''

import os
import numpy as np
import cv2
import csv

# Constants to pull the data from
ROOT_DIR = 'training_data'
LOG = 'driving_log.csv'
IMG_DIR = 'IMG'

# Unpack the row into it's values.
def unpack_row(row):
    out = {}
    out['center'] = row[0]
    out['left'] = row[1]
    out['right'] = row[2]
    out['angle'] = row[3]
    out['throttle'] = row[4]
    out['break'] = row[5]
    out['speed'] = row[6]
    return out

# Using the CSV, pull the data and structure it. We can use opencv to load
# the images into arrays.
def get_data():
    # Create an iterator to go over the log.
    driving_log = os.path.join(ROOT_DIR, LOG)
    iterator = csv.reader(open(driving_log))

    # Initialize an angle correction value for left and right images.
    correction = 0.2

    # Create containers to hold our X_train, and y_train data.
    X_train, y_train = [], []

    # Iterate over our data.
    for row in iterator:
        # Skip the header row.
        if 'throttle' in row:
            continue

        # Unpack the data into our row dict.
        row = unpack_row(row)

        # The value we're trying to predict is the steering angle. Extract the
        # 3 images, and compute the left and right angles too.
        read_image = lambda key: cv2.imread(os.path.join(ROOT_DIR, row[key].strip()))
        left, center, right = read_image('left'), read_image('center'), read_image('right')
        center_angle = float(row['angle'])
        left_angle = center_angle + correction
        right_angle = center_angle - correction

        # Augment the data with some inversions. The angle of the inverted
        # left image is -1 * the right angle, and vice versa.
        left_flipped = np.fliplr(left)
        right_flipped = np.fliplr(right)
        left_flipped_angle = -1 * left_angle
        right_flipped_angle = -1 * right_angle

        # Combine all our angles and steering measurements together.
        images = [left, left_flipped, right, right_flipped, center]
        angles = [left_angle, left_flipped_angle, right_angle,
                  right_flipped_angle, center_angle]

        # Extend data to training set.
        X_train.extend(images)
        y_train.extend(angles)

    # Return the X_train and y_train.
    return np.array(X_train), np.array(y_train)
