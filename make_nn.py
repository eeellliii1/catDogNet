import tensorflow as tf
import numpy as np
import os
import sys
import cv2

from tensorflow.keras import layers, models
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam

# Get image transformations for data augmentation
# Ex: flipped vertically, flipped horizontally, flipped vertically and horizontally
def get_img_variations(img):
    return [img, np.flip(img, 0), np.flip(img, 1), np.flip(np.flip(img, 1), 0)]

# Read in all images in the given directory as pixel arrays
def import_image_data(dir_name):
    # Get the filenames of all images in dir
    pic_ls = os.listdir(dir_name)

    # Store the image pixels and the labels for those images
    pic_data = []
    label_data = []

    counter = 0
    every_x = 1 # THIS IS USED SOLELY FOR TRAINING EXPERIMENTATION ex: every 5 images or every 3 images would be selected to train on 
    
    # Process each image
    for i in range(len(pic_ls)):
        pic = pic_ls[i]

        if not counter % every_x == 0:  
            counter += 1
            continue                    
        counter += 1                    

        # Make sure we are looking at jpg files (maybe should add png files but for this assignment its ok probably)        
        if not pic.endswith(".jpg"):
            continue

        # Convert image at filepath to pixel array
        pixel_inf = cv2.resize(cv2.imread(dir_name + '/' + pic), (100, 100))

        # Get transformations and add to lists along with label
        pic_data_trans = get_img_variations(pixel_inf)
        for pic_trans_inf in pic_data_trans:
            pic_data.append(pic_trans_inf)    
            if pic[0] == 'c':
                label_data.append([1, 0])
            else:
                label_data.append([0, 1])

    # Make arrays into np arrays so they can be used by model
    pic_data = np.array(pic_data)
    label_data = np.array(label_data)

    return pic_data, label_data

# Generate and return our ready-to-be-trained model
def generate_model():
    # Hyperparams
    img_dim = 100
    filter_dim = 3
    pool_dim = 2
    color_pix_dim = 3
    activation = "relu"
    drop_prop = 0.5

    # Make the model
    model = models.Sequential()

    # Add layers of the following format
    num_layers = 4
    for i in range(num_layers):
        model.add(layers.Conv2D(img_dim, (filter_dim, filter_dim), activation = activation, input_shape = (img_dim, img_dim, color_pix_dim)))
        model.add(layers.BatchNormalization()) # Good to be after activation func layer
        model.add(layers.MaxPooling2D((pool_dim, pool_dim)))
        model.add(layers.Dropout(drop_prop)) # To prevent info bottleneck

    model.add(layers.Flatten())

    dense_nodes = 256
    model.add(layers.Dense(dense_nodes, activation = activation))
    
    outputs = 2
    model.add(layers.Dense(outputs, activation = "softmax")) # Output layer

    # Display model summary
    print(model.summary())
        
    # Compile model with adam (not a smith) optimizer
    learning_rate = 0.001
    optimizer = Adam(learning_rate = learning_rate)
    loss = 'binary_crossentropy'
    metrics = ["accuracy"]
    model.compile(optimizer = optimizer, loss = loss, metrics = metrics)
        
    return model


######################################################################
# Command line in format of: <img_sources> <network_name>

# Get args from the user
_, dir_name, net_name = sys.argv

# Load the image data
print("--Loading IMG Data")
x_train, y_train = import_image_data(dir_name)

# Generate the model
print("--Generating Model")
model = generate_model()

# Train the model
print("--Training Model")
epochs = 45
batch_size = 24
model.fit(x_train, y_train, batch_size = batch_size, epochs = epochs)

# Save the model
print("--Saving Model")
model.save(net_name, save_format = 'h5', include_optimizer = False)
