import tensorflow as tf
import numpy as np
import os
import sys
import cv2

# Reads in the given pictures as pixel arrays
def read_pic_data(pic_paths):
    pixel_data = []
    for pic in pic_paths:
        p_dat = cv2.imread(pic)

        pixel_data.append(p_dat)

    return np.array(pixel_data)

# Print out the name of each picture and it was classified as cat or dog
def display_results(pics, labels):
    for i in range(len(pics)):
        surety = labels[i][1]
        animal = "dog"
        if labels[i][0] > labels[i][1]:
            surety = labels[i][0]
            animal = "cat"

        print(pics[i], "contains a", animal, surety)

def load_images_from_dir(dir_name):
    img_paths = []

    # Get the filenames of all images in dir
    pic_ls = os.listdir(dir_name)

    for pic_name in pic_ls:
        img_paths.append(dir_name + '/' + pic_name)

    return img_paths

# Get the user args
args = sys.argv

# Get the path to the neural net
nn_path = args[1]

pic_paths = []

if args[2] == 'ls':
    # Every further argument should be an image to check
    pic_paths = args[3:]
elif args[2] == 'dir':
    pic_paths = load_images_from_dir(args[3])

# Process images for prediction
input_data = read_pic_data(pic_paths)

# Load saved model
model = tf.keras.models.load_model(nn_path)

# Get predicted labels
labels = model.predict(input_data)

# Show the classification results
display_results(pic_paths, labels)