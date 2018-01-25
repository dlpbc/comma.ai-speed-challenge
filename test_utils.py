import sys
import cv2
import numpy as np

'''
Generate test examples from video data.
Each example generated is a sequence of video frames (clip).

Parameters:
    video_file: the video file to read from and generate test data.
    num_frames_per_example: number of video frames each generated example should contain.
    batch_size: size of batch to generate each time generated yields examples.
    two_channels_frame: bool. if True, frames in examples will contain only 2 channels. Otherwise,
                        frames will contain the normal 3 channels. This option was added because
                        the pretrained model from 'Quo Vardis paper: Kinetics Inflated 3d Inception
                        architecture' uses 2 channel frames for optical flow.

Returns:
    X_batch test data of `batch_size`
'''
def generate_test_data(video_file, num_frames_per_example=40, batch_size=32, two_channels_frame=False):
    X_batch = []

    # load video
    cap = cv2.VideoCapture(video_file)

    frames = []
    while True:
        ret, frame = cap.read()
        if frame is None:
            # yield the number examples we currently have
            # even though it's not up to the number examples
            # in a normal batch of `batch_size`
            yield np.array(X_batch)
            # break out from loop
            break

        # original shape of frame 640 width, 480 height
        # determine number of channels for frames
        if two_channels_frame:
            frame = frame[..., 1 : ] # slicing off the 'B' channel in 'BGR'

        # preprocess frame
        frame = _preprocess_frame(frame)

        frames.append(frame)

        if len(frames) == num_frames_per_example:
            X_batch.append(np.array(frames))

            if len(X_batch) == batch_size:
                yield np.array(X_batch)
                X_batch = [] # reset for new batch

            # update frame list by removing the first frame in the list
            # next example will be based on the 2nd - 39th frame and the
            # new frame that will be read.
            frames = frames[1 : ]

    cap.release()
    return


'''
Preprocess test video frame similar to the way the training and validation frames 
were preprocessed.

Argument: 
    frame: the frame to preprocess

Return:
    frame: the preprocessed frame
'''
def _preprocess_frame(frame):
    # original frame size 640 (width) x 480 (height)

    # crop out 256 x 256 in a centered region of the frame
    h_center_pixel_idx = frame.shape[0] // 2
    w_center_pixel_idx = frame.shape[1] // 2
    h_start_idx = h_center_pixel_idx - 128
    h_stop_idx = h_center_pixel_idx + 128
    w_start_idx = w_center_pixel_idx - 128
    w_stop_idx = w_center_pixel_idx + 128
    frame = frame[h_start_idx : h_stop_idx, w_start_idx : w_stop_idx]

    # resize frame to size 70 x 70
    frame = cv2.resize(frame, (70, 70))

    # cropout 64 x 64 in a centered region of the frame
    h_center_pixel_idx = frame.shape[0] // 2
    w_center_pixel_idx = frame.shape[1] // 2
    h_start_idx = h_center_pixel_idx - 32
    h_stop_idx = h_center_pixel_idx + 32
    w_start_idx = w_center_pixel_idx - 32
    w_stop_idx = w_center_pixel_idx + 32
    frame = frame[h_start_idx : h_stop_idx, w_start_idx : w_stop_idx]


    # scale pixels of frame to values between -1 and 1
    frame = (np.float32(frame) / 127.5) - 1.0 

    return frame # size 64 x 64
