import os
import sys
import cv2
import numpy as np

from keras.models import Sequential
from keras.layers import InputLayer
from keras.layers import Conv3D
from keras.layers import Reshape
from keras.layers import Lambda
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Dropout

from i3d.i3d_inception import Inception_Inflated3d

'''
builds the model architecture
Arguments:
    num_input_frames: number of frames per example
    frame_height: height of frame
    frame_width: width of frame
    num_frame_channels: number of channels in input frames
    base_model_weights: pretrained weights for base model. When set to None,
                        pretrained weights is not loaded
    freeze_base: bool. whether or not to freeze the base model
'''
def build_model(num_input_frames, frame_height, frame_width, num_frame_channels, base_model_weights=None, freeze_base=False):

    base_model = Inception_Inflated3d(
        include_top=False,
        weights=base_model_weights,
        input_shape=(num_input_frames, frame_height, frame_width, num_frame_channels),
        classes=None)

    input_shape = (num_input_frames, frame_height, frame_width, num_frame_channels)
    input_layer = InputLayer(input_shape=input_shape)

    model = Sequential()
    model.add(input_layer)
    model.add(base_model)
    # shape (None, 4, 1, 1, 1024) when num frames per example = 40

    base_model_output_frames = model.output_shape[1]
    base_model_output_channels = model.output_shape[4]
    model.add(Reshape((base_model_output_frames, base_model_output_channels)))
    model.add(Lambda(lambda x: K.mean(x, axis=1, keepdims=False), output_shape=lambda s: (s[0], s[2])))
    # shape (None, 1024) 
    model.add(Dense(512, activation='relu', kernel_initializer='he_normal'))
    model.add(Dropout(0.5))

    model.add(Dense(256, activation='relu', kernel_initializer='he_normal'))
    model.add(Dropout(0.5))

    model.add(Dense(NUM_OUTPUT))

    return model
 


'''
Python Generator for training or validation examples (dependent on specified `data_path`)

Arguments
    data_path: path to the directory containing batch files (`*.npz files) of train or validation
               data
    batch_size: size of batch to be generated
    shuffle: shuffle the order generated examples in the batch before returning it.
             helps in network generalization during training because we prevent the network from
             memorizing the default order in which examples.
    center_crop: specify whether frame/image crop region should be centered. cropping is a data
                 augmentation step produce examples of shape 64 x 64 from a 106 x 80 (width x height)
                 frame shape. if `crop_crop` is False, the crop region is randomly determined. 
                 otherwise, a crop region centered on the frames/images is used.
    two_channels_frame: bool. if True, frames in examples will contain only 2 channels. Otherwise,
                        frames will contain the normal 3 channels. This option was added because
                        the pretrained model from 'Quo Vardis paper: Kinetics Inflated 3d Inception
                        architecture' uses 2 channel frames for optical flow.
Return
    [X_batch, y_batch]: returns `batch_size` number of examples and corresponding labels
'''
def video_data_generator(data_path, batch_size=32, shuffle=False, center_crop=False, two_channels_frame=False):
    final_frame_size = (64, 64) # the height and width of frames in example

    # get number of `*npz` files in `data_path`
    num_data_files = len(os.listdir(data_path))

    curr_file_idx = 1
    X_data = y_data = None
    X_batch = y_batch = None
    load_next_file = True

    while True:
        if load_next_file:
            
            # load npz file
            data_file = np.load('{0}/batch_{1}.npz'.format(data_path, curr_file_idx))
            X_data = data_file['examples']
            y_data = data_file['labels']

            load_next_file = False

        if X_data.shape[0] < batch_size: # we have to load a new data file

            load_next_file = True

            # determine the next index of data file to laod
            # `curr_file_idx` should always be less than or
            # equal to `num_data_files`
            if curr_file_idx == num_data_files:
                curr_file_idx = 1 # reset back to the beginning

            else: # it is less than `num_data_files`
                curr_file_idx += 1 # increment

            continue

        X_batch = X_data[ : batch_size]
        y_batch = y_data[ : batch_size]

        # update the X_data and y_data to reflect what's left
        X_data = X_data[batch_size : ]
        y_data = y_data[batch_size : ]

        # determine number of channels for frames
        if two_channels_frame:
            X_batch = X_batch[..., 1 : ] # slicing off the 'B' channel in 'BGR'
            # now we are left with just two channels
            assert X_batch.shape == (batch_size, 40, 70, 70, 2)
        else:
            assert X_batch.shape == (batch_size, 40, 70, 70, 3)

        # PREPROCESSING:
        # scale pixels of frame to values between -1 and 1
        X_batch = (np.float32(X_batch) / 127.5) - 1.0 # ensures the returned tensor is float32 type

        # DATA AUGMENTATION:
        # select a 64x64 crop area from the current examples of size 70x70
        curr_h, curr_w = X_batch[0].shape[1:3]

        if center_crop: # crop a 64 x 64 portion that is centered from the frame
            h_center_pixel_idx = curr_h // 2
            w_center_pixel_idx = curr_w // 2

            h_start_idx = h_center_pixel_idx - (final_frame_size[0] // 2)
            h_stop_idx = h_center_pixel_idx + (final_frame_size[0] // 2)
            w_start_idx = w_center_pixel_idx - (final_frame_size[1] // 2)
            w_stop_idx = w_center_pixel_idx + (final_frame_size[1] // 2)

        else: # randomly crop 64 x 64 portion from frame 
            # max indx to start crop that will not be out-of-bound
            height_threshold = curr_h - final_frame_size[0] 
            # max index to start crop that will not be out-of-bound
            width_threshold = curr_w - final_frame_size[1] 

            h_start_idx = np.random.randint(low=0, high=height_threshold)
            h_stop_idx = h_start_idx + final_frame_size[0]
            w_start_idx = np.random.randint(low=0, high=width_threshold)
            w_stop_idx = w_start_idx + final_frame_size[1]
            
        X_batch = X_batch[ : , : , h_start_idx : h_stop_idx, w_start_idx : w_stop_idx]

        if shuffle:
            indexes = np.arange(batch_size)
            np.random.shuffle(indexes)
            X_batch = X_batch[indexes]
            y_batch = y_batch[indexes]

        yield X_batch, y_batch # batch completed, return result

    return

'''
compute the total number examples in all 'npz' files in `data_path`
Argument:
    data_path: directory path where npz files are stored

Returns:
    total number of examples
'''
def get_total_num_examples(data_path):
    total_num = 0
    num_data_files = len(os.listdir(data_path))

    for i in np.arange(1, num_data_files+1):
        # load npz file
        data_file = np.load('{0}/batch_{1}.npz'.format(data_path, i))
        total_num += len(data_file['labels'])

    return total_num


