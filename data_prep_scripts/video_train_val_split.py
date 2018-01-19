import os
import shutil
import argparse
import cv2


'''
splits a video file into two videos (one becomes training data video and the other validation data)
based on train/validation split ratio specified.
e.g. if split ratio is 0.7, then the train video is made up of the first 70% frames,
     while the validation video becomes the remaining 30% frames.

Args
    video_path: path to the video file to split
    label_path: path to the label file to split (it contains speed at each frame)
    output_dir: directory to store output
    frame_size: size of each frame. a tuple -> (width, height)
    fps: frame rate (frame per second)
    split_ratio: train/validation split ratio

Return: None

But the function creates 4 files if no error ocurred during split. These file are listed below:
1. training split video (containing frames from beginning of `video_path` to split point)
2. training split label (containing labels from beginning of `label_path to split point)
3. validation split video (containing frames from split point to the end of `video_path`)
4. validation split label (containing labels from split ponit to the end of `label_path`)

Note: total number of frames in `video_path` should be equal to the number of lines in `label_path`

'''
def train_val_split(video_path, label_path, output_dir, frame_size, fps, split_ratio=0.7):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    os.makedirs(output_dir)

    # variables
    train_split_video_path = output_dir + 'train_split_video.mp4'
    train_split_label_path = output_dir + 'train_split_label.txt'
    validation_split_video_path = output_dir + 'validation_split_video.mp4'
    validation_split_label_path = output_dir + 'validation_split_label.txt'

    # open video to split
    video_data_reader = cv2.VideoCapture(video_path)

    # open the corresponding label file to split
    label_data_reader = open(label_path, 'r')

    # get total number of video_path frames
    total_num_frames = int(video_data_reader.get(cv2.CAP_PROP_FRAME_COUNT))

    # sanity check
    if len(label_data_reader.readlines()) != total_num_frames:
        print('ERROR: number of frames and number of labels should be thesame')
        print('exiting')
        return

    label_data_reader.seek(0) # reset file pointer position back to the beginning
        

    # specify a video compression format (codec)
    video_compression_codec = cv2.VideoWriter_fourcc(*'XVID') # output moderate size

    # create video file and label file
    # to store frames and corresponding labels of training data split
    train_split_video_writer = cv2.VideoWriter(train_split_video_path,
        video_compression_codec,
        fps,
        frame_size)
    train_split_label_writer = open(train_split_label_path, 'w')

    # create video file and label file
    # to store frames and corresponding labels of validation data split
    validation_split_video_writer = cv2.VideoWriter(validation_split_video_path,
        video_compression_codec,
        fps,
        frame_size)
    validation_split_label_writer = open(validation_split_label_path, 'w')


    error_occurred = False
    idx_end_train = int(total_num_frames * split_ratio)
    # copy the required number of frames into train split video file
    # i.e. from frame 1 to (split_ratio * total_num_frames)
    for i in range(0, idx_end_train):
        ret, frame = video_data_reader.read()

        if frame is None: # this should not happen if arg total_num_frame is specified correctly
            print('an error occurred while trying to read frames')
            print('make sure the argument `total_num_frames is correct')
            print('exiting')
            error_occurred = True
            break

        # save frame to train split video file
        train_split_video_writer.write(frame)
        # save corresponding frame label to train split label file
        label = label_data_reader.readline()
        train_split_label_writer.write(label)

    if not error_occurred:
        # copy the required number of frames into validation split video file
        # i.e. from frame (split_ratio * total_num_frames) + 1 to the end
        for i in range(idx_end_train, total_num_frames):
            ret, frame = video_data_reader.read()

            if frame is None: # this should not happen if arg total_num_frame is specified correctly
                print('an error occurred while trying to read frames')
                print('make sure the argument `total_num_frames is correct')
                print('exiting')
                error_occurred = True
                break

            # save frame to validation split video file
            validation_split_video_writer.write(frame)
            # save corresponding frame label to validation split label file
            label = label_data_reader.readline()
            validation_split_label_writer.write(label)


    # close opened resources
    video_data_reader.release()
    train_split_video_writer.release()
    validation_split_video_writer.release()

    label_data_reader.close()
    train_split_label_writer.close()
    validation_split_label_writer.close()

    if error_occurred:
        # error occured during split, delete created files
        os.remove(train_split_video_path)
        os.remove(train_split_label_path)
        os.remove(validation_split_video_path)
        os.remove(validation_split_label_path)
    else:
        print('Files created: ')
        print(train_split_video_path)
        print(train_split_label_path)
        print(validation_split_video_path)
        print(validation_split_label_path)
   

    return

def main(args):
    train_val_split(args.video_path, args.label_path, args.output_dir, (640, 480), 20.0, 0.7)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('video_path',
                        help='path to video to split into train video and validation video.',
                        type=str)
    parser.add_argument('label_path',
                        help='path to label file to split into train video and validation labels.',
                        type=str)
    parser.add_argument('output_dir',
                        help='path to output train split and validation split videos and labels.',
                        type=str)
    main(parser.parse_args())
