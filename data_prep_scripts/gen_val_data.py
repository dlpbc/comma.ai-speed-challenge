'''
This script does 2 things
1. generate example clip (each example contains 40 70x70 frames) from train video and stores a bunch of 
   this clips in 'npz' files as batches. A typical npz batch file contains 1024 examples.
2. for each example generated, a duplicate copy may be created or the example might not be saved
   because of the reason stated below.

REASON: the current form of the train video has a frame labels skewed towards speeds (labels) of a 
        certain category of speed than others (e.g. labels for speed between 0 - 5 are more than labels
        for speed > 25).
        In order to balance this and at least provide some balance to enable mean square error learn
        appropriately, we use this script to generate train batches (npz files) which contains a more balanced
        training examples (i.e. balanced across speed categories). This is done by eliminating some generated
        example of a certain speed range category (to reduce its number) and then increase/duplicating examples of a
        some speed range category (to increase its number)
        
'''
import os
import shutil
import sys
import argparse
import numpy as np
import cv2


NUM_FRAMES_PER_EXAMPLE = 40
STRIDE = 1
BATCH_SIZE = 1024

def main(args):
    label_path = args.label_path
    video_path = args.video_path
    output_dir = args.output_dir

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    os.makedirs(output_dir)


    # open label file
    f = open(label_path, 'r')
    labels = f.readlines()
    f.close()

    # open train split video file
    cap = cv2.VideoCapture(video_path)
    total_num_video_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    # sanity check
    assert len(labels) == total_num_video_frames
        

    batch_examples = []
    batch_labels = []
    frames = []
    idx = 0
    batch_num = 0
    gen_num_examples = 0

    # loop thru frames in video
    while True:
        ret, frame = cap.read()

        if frame is None:
            break

        # original size of frame 640 x 480 (width x height)

        # crop out 256 x 256 in a centered region of the frame
        h_center_pixel_idx = frame.shape[0] // 2
        w_center_pixel_idx = frame.shape[1] // 2

        h_start_idx = h_center_pixel_idx - 128
        h_stop_idx = h_center_pixel_idx + 128
        w_start_idx = w_center_pixel_idx - 128
        w_stop_idx = w_center_pixel_idx + 128

        frame = frame[h_start_idx : h_stop_idx, w_start_idx : w_stop_idx]
        assert frame.shape[:2] == (256, 256)

        # resize the frame to 70 x 70
        frame = cv2.resize(frame, (70, 70))

        # store frame
        frames.append(frame)

        if len(frames) == NUM_FRAMES_PER_EXAMPLE:
            # an example to be generated
            
            # read label
            label = float(labels[idx])

            # generate example
            _frames = np.array(frames, copy=True)
            batch_examples.append(_frames)
            batch_labels.append(label)
            gen_num_examples += 1
     
            if len(batch_examples) >= BATCH_SIZE: # one batch is completed
                # save batch batch_examples to disk
                np.savez(output_dir + 'batch_{0}.npz'.format(batch_num+1), 
                   examples=np.asarray(batch_examples[ : BATCH_SIZE]), 
                   labels=np.expand_dims(np.asarray(batch_labels[ : BATCH_SIZE], dtype=np.float32), 1))

                # update batch_examples and label list
                batch_examples = batch_examples[BATCH_SIZE : ]
                batch_labels = batch_labels[BATCH_SIZE : ]

                # update batch number
                batch_num += 1
                
            # now focus on generating the next example
            frames = frames[STRIDE : ]
        # end of if len(frames) == NUM_FRAMES_PER_EXAMPLE

        idx += 1
        sys.stdout.write('\rNum of batch_examples generated: %d of %d' % gen_num_examples)

    # end of while True

    if len(batch_examples) > 0:
        # no more frames to read
        # but there are some batch_examples in the list to save
        np.savez(output_dir + 'batch_{0}.npz'.format(batch_num+1), 
                 examples=np.asarray(batch_examples),
                 labels=np.expand_dims(np.asarray(batch_labels, dtype=np.float32), 1))

        batch_num += 1
        sys.stdout.write('\rNum of batch_examples generated: %d' % gen_num_examples)

    # close video file
    cap.release()

    print('\ncompleted...')
    print('number of batch_examples generated: %d' % gen_num_examples)
    print('total number of batches saved to disk: %d' % batch_num)

    return

if __name__ == '__main__':
    print('\ngenerating validation example clips from validation video split\n')
    parser = argparse.ArgumentParser()
    parser.add_argument('video_path',
                        help='path to the video from which to generate example clips',
                        type=str)
    parser.add_argument('label_path',
                        help='path to the label file for corresponding video',
                        type=str)
    parser.add_argument('output_dir',
                        help='path to store output (npz files)',
                        type=str)

    main(parser.parse_args())
