# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

import keras
from keras import layers
from keras.callbacks import ReduceLROnPlateau

# import io
import os
import h5py

import numpy as np
import tensorflow as tf
import glob
# import torch
# from torchsummary import summary
from PIL import Image


radar_data_folder_path = '../RadarData/'  # Replace this with your folder path
# Search for all files with .txt extension in the folder
radar_files = glob.glob(os.path.join(radar_data_folder_path, '*.scu'))

input_files = []
# read each file in each folder from folder_path
import os

# Walk through all directories and files
for radar_folders in sorted(os.listdir(radar_data_folder_path)):
    # Construct the full path to the folder
    folder_path = os.path.join(radar_data_folder_path, radar_folders)

    # Check if the path is a directory
    if os.path.isdir(folder_path):
        print(f"\nProcessing folder: {folder_path}")
        # use glob.glob to read all files in the folder order by name
        radar_files = sorted(glob.glob(os.path.join(folder_path, '*.scu')))
        # Walk through all directories and files inside the current folder
        radar_day_file = []
        # Process each file in the current directory
        for radar_file in radar_files:
            # Construct the full path to the file
            # radar_file = os.path.join(folder_path, filename)

            with h5py.File(radar_file, 'a') as file:
                a_group_key = list(file.keys())[0]
                dataset_DXk = file.get(a_group_key)
                ds_arr = dataset_DXk.get('image')[:]  # the image data in an array of floats
                ds_arr = np.where(ds_arr == -999, 0, ds_arr)
                # Convert the 2D array to a PIL Image
                image = Image.fromarray((ds_arr * 255).astype(np.uint8))
                resized_image = image.resize((128, 128))
                # plt.imshow(resized_image)
                # plt.show()
                # plt.imshow(image)
                # plt.show()
                # Convert the resized image back to a 2D NumPy array
                resized_image_array = np.array(resized_image)

                radar_day_file.append(resized_image_array)
                file.close()
        input_files.append(radar_day_file)

input_files = np.stack(input_files, axis=0)

input_files = np.expand_dims(input_files, axis=-1)
dataset = input_files


# Split into train and validation sets using indexing to optimize memory.
indexes = np.arange(dataset.shape[0])
np.random.shuffle(indexes)
train_index = indexes[: int(0.9 * dataset.shape[0])]
val_index = indexes[int(0.9 * dataset.shape[0]):]
train_dataset = dataset[train_index]
val_dataset = dataset[val_index]


# Normalize the data to the 0-1 range.
# train_dataset = train_dataset / 255
# val_dataset = val_dataset / 255

# We'll define a helper function to shift the frames, where
# `x` is frames 0 to n - 1, and `y` is frames 1 to n.
def create_shifted_frames(data):
    x = data[:, 0: data.shape[1] - 1, :, :]
    y = data[:, 1: data.shape[1], :, :]
    return x, y


# Apply the processing function to the datasets.
x_train, y_train = create_shifted_frames(train_dataset)
x_val, y_val = create_shifted_frames(val_dataset)

# Inspect the dataset.
print("Training Dataset Shapes: " + str(x_train.shape) + ", " + str(y_train.shape))
print("Validation Dataset Shapes: " + str(x_val.shape) + ", " + str(y_val.shape))

# input_shape = (5, 256, 256, 1)  # (time_steps, height, width, channels)
inp = layers.Input(shape=(None, *x_train.shape[2:]))

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# We will construct 3 `ConvLSTM2D` layers with batch normalization,
# followed by a `Conv3D` layer for the spatiotemporal outputs.
x = layers.ConvLSTM2D(
    filters=16,
    kernel_size=(5, 5),
    padding="same",
    return_sequences=True,
    activation="relu",
)(inp)
x = layers.BatchNormalization()(x)
x = layers.ConvLSTM2D(
    filters=64,
    kernel_size=(3, 3),
    padding="same",
    return_sequences=True,
    activation="relu",
)(x)
x = layers.BatchNormalization()(x)
x = layers.ConvLSTM2D(
    filters=64,
    kernel_size=(1, 1),
    padding="same",
    return_sequences=True,
    activation="relu",
)(x)
x = layers.Conv3D(
    filters=1, kernel_size=(3, 3, 3), activation="sigmoid", padding="same"
)(x)
# check if GPU device exists
print("GPUs Available: ", tf.config.experimental.list_physical_devices('GPU'))

# Construct the model.
with tf.device("/GPU:0"):
    model = keras.models.Model(inp, x)
    model.compile(
        loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.Adam(),
    )

    # Define some callbacks to improve training.
    #early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=5)

    # Define modifiable training hyperparameters.
    epochs = 20
    batch_size = 2

    # Fit the model to the training data.
    model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_val, y_val),
        #callbacks=[early_stopping, reduce_lr],
        callbacks=[reduce_lr],
    )

# # Construct the input layer with no definite frame size.
# #inp = layers.Input(shape=(None, *input_shape[1:]))
# inp = layers.Input(shape=(None, *x_train.shape[2:]))
#

#
#
# # Next, we will build the complete model and compile it.
# model = keras.models.Model(inp, x)
# model.compile(
#     loss=keras.losses.binary_crossentropy,
#     optimizer=keras.optimizers.Adam(),
# )

#
# # Define some callbacks to improve training.
# early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)
# reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=5)
#
# # Define modifiable training hyperparameters.
# epochs = 20
# batch_size = 5
#
# # Fit the model to the training data.
# model.fit(
#     x_train,
#     y_train,
#     batch_size=batch_size,
#     epochs=epochs,
#     validation_data=(x_val, y_val),
#     callbacks=[early_stopping, reduce_lr],
# )
