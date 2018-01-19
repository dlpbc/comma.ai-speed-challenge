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
        

    # statistics of each category
    # this is especially relevant to categories with decimal scale factor (e.g '2.3', '1.5' etc)
    # for example '2.3' means that 
    # for every 10 consecutive (collection of frames ready to be used for example generation) 
    # of the same label type (category),
    # 3 of those frames collection will be used to produce 3 cropped batch_examples
    # while 7 of those frame collections will be used to produce 2 cropped batch_examples

    # using '2.3' scale factor as an example,
    # the FIRST ELEMENT of the list stores the number of frame collections (with the same) label type 
    # that has been parsed. This should not exceed 10, after which we reset back to 0
    # the SECOND ELEMENT of the list stores the number of 'third' batch_examples that have been generated
    # so far 
    cat_stats = {
        '0-2'   : [0,0], 
        '3-4'   : [0,0],
        '5-6'   : [0,0],
        '7-8'   : [0,0],
        '9-10'  : [0,0],
        '11-12' : [0,0],
        '13-14' : [0,0],
        '15-16' : [0,0],
        '17-18' : [0,0],
        '19-20' : [0,0],
        '21-22' : [0,0],
        '23-24' : [0,0],
        '25-26' : [0,0],
        '27-28' : [0,0]
    }

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
            int_label = int(label)

            # determine example category based on label
            if int_label >= 0  and int_label <=  2: 
                # scale up, factor 2.3

                # generate first example
                _frames = np.array(frames, copy=True)
                batch_examples.append(_frames)
                batch_labels.append(label)
                gen_num_examples += 1

                # generate second example
                _frames = np.array(frames, copy=True)
                batch_examples.append(_frames)
                batch_labels.append(label)
                gen_num_examples += 1

                # sanity check
                if cat_stats['0-2'][0] > 10: 
                    print('\n\nSomething went wrong some where')
                    print('Category: 0-2')
                    print('iteration: %d' % idx)
                    print('Parsed example count > 10')
                    print('forgot to reset count somewhere')
                    print('Exiting...\n\n')
                    sys.exit()

                # update stat to denote that we have parsed through another frame collection
                # of this category (regardless of the outcome of probability of generating a 
                # 3rd example from it)
                cat_stats['0-2'][0] += 1

                # generate third example or not based on probability

                # use probablity to randomly determine if we are going to generate a third
                # example using this example.
                # the '.3' in '2.3' scale factor means for a every 10 batch_examples with this same
                # label value, we generate a third example
                if cat_stats['0-2'][0] > 7 and cat_stats['0-2'][1] < 3:
                    # this means that if we have already parsed 7 frame collections
                    # and we have not yet generated a third example from the 7 previous frame collections
                    # parsed.
                    # we no longer have the luxury of probability, just generate a third example
                    # for this frame collection and the next 2 future frame collections in the same
                    # category.

                    # generate another(third) example
                    _frames = np.array(frames, copy=True)
                    batch_examples.append(_frames)
                    batch_labels.append(label)
                    gen_num_examples += 1

                    # update the stat to reflect the current number of third example generated
                    cat_stats['0-2'][1] += 1

                elif cat_stats['0-2'][0] <= 6 and cat_stats['0-2'][1] < 3:
                    # we still have the luxury to flip a coin (random probability)
                    # to see if we should generate a third batch_examples using
                    # this frame collection, or push the responsiblity to a future frame collection
                    # of the same category (same label)

                    # flip a coin
                    outcome = np.random.binomial(1, 0.5)
                    # postive outcome, generate 'third' example
                    if outcome == 1:
                        # generate another(third) example
                        _frames = np.array(frames, copy=True)
                        batch_examples.append(_frames)
                        batch_labels.append(label)
                        gen_num_examples += 1

                        # update the stat to reflect the current number of third example generated
                        cat_stats['0-2'][1] += 1
                    else:
                        # negative outcome
                        # do nothing
                        pass

                # sanity check
                if cat_stats['0-2'][0] == 10 and cat_stats['0-2'][1] != 3:
                    print('\n\nSomething went wrong some where')
                    print('Category: 0-2')
                    print('iteration: %d' % idx)
                    print('Exiting...\n\n')
                    sys.exit()

                if cat_stats['0-2'][0] == 10:
                    # reset stats for this category
                    cat_stats['0-2'][0] = 0
                    cat_stats['0-2'][1] = 0
            # end of if label >= 0  and label <=  2: 
                    
      
            if int_label >= 3  and int_label <=  4: 
                # scale up, factor 1.7

                # generate first example
                _frames = np.array(frames, copy=True)
                batch_examples.append(_frames)
                batch_labels.append(label)
                gen_num_examples += 1

                # sanity check
                if cat_stats['3-4'][0] > 10: 
                    print('\n\nSomething went wrong some where')
                    print('Category: 3-4')
                    print('iteration: %d' % idx)
                    print('Parsed example count > 10')
                    print('forgot to reset count somewhere')
                    print('Exiting...\n\n')
                    sys.exit()

                # update stat to denote that we have parsed through another frame collection
                # of this category (regardless of whether we generated a 2nd example from 
                # it or not)
                cat_stats['3-4'][0] += 1

                # generate second example or not based on probability

                # use probablity to randomly determine if we are going to generate a second
                # example using this frame collection.
                # the '.7' in '1.7' scale factor means for a every 10 batch_examples with this same
                # label value, we generate a second example
                if cat_stats['3-4'][0] > 3 and cat_stats['3-4'][1] < 7:
                    # this means that if we have already parsed 3 frame collections
                    # and we have not yet generated a second example from the 3 previous 
                    # frame collections parsed.
                    # we no longer have the luxury of probability, just generate a second example
                    # for this frame collection and the next 6 future frame collections in the same
                    # category.

                    # generate another(second) example
                    _frames = np.array(frames, copy=True)
                    batch_examples.append(_frames)
                    batch_labels.append(label)
                    gen_num_examples += 1

                    # update the stat to reflect the current number of second example generated
                    cat_stats['3-4'][1] += 1

                elif cat_stats['3-4'][0] <= 2 and cat_stats['3-4'][1] < 7:
                    # we still have the luxury to flip a coin (random probability)
                    # to see if we should generate a second batch_examples using
                    # this frame collection, or push the responsiblity to a future frame collection
                    # of the same category (same label)

                    # flip a coin
                    outcome = np.random.binomial(1, 0.5)
                    # postive outcome, generate 'second' example
                    if outcome == 1:
                        # generate another(second) example
                        _frames = np.array(frames, copy=True)
                        batch_examples.append(_frames)
                        batch_labels.append(label)
                        gen_num_examples += 1

                        # update the stat to reflect the current number of second example generated
                        cat_stats['3-4'][1] += 1
                    else:
                        # negative outcome
                        # do nothing
                        pass

                # sanity check
                if cat_stats['3-4'][0] == 10 and cat_stats['3-4'][1] != 7:
                    print('\n\nSomething went wrong some where')
                    print('Category: 3-4')
                    print('iteration: %d' % idx)
                    print('Exiting...\n\n')
                    sys.exit()

                if cat_stats['3-4'][0] == 10:
                    # reset stats for this category
                    cat_stats['3-4'][0] = 0
                    cat_stats['3-4'][1] = 0
            # end of if label >= 3  and label <= 4: 
     
            if int_label >= 5  and int_label <=  6: 
                # scale up, factor 1.5
                _frames = np.array(frames, copy=True)
                batch_examples.append(_frames)
                batch_labels.append(label)
                gen_num_examples += 1

                # sanity check
                if cat_stats['5-6'][0] > 10: 
                    print('\n\nSomething went wrong some where')
                    print('Category: 5-6')
                    print('iteration: %d' % idx)
                    print('Parsed example count > 10')
                    print('forgot to reset count somewhere')
                    print('Exiting...\n\n')
                    sys.exit()

                # update stat to denote that we have parsed through another frame collection
                # of this category (regardless of whether we generated a 2nd example from 
                # it or not)
                cat_stats['5-6'][0] += 1

                # use probablity to randomly determine if we are going to generate a second
                # example using this frame collection.
                # the '.5' in '1.5' scale factor means for a every 10 batch_examples with this same
                # label value, we generate a second example
                if cat_stats['5-6'][0] > 5 and cat_stats['5-6'][1] < 5:
                    # this means that if we have already parsed 5 frame collections
                    # and we have not yet generated a second example from the 5 previous 
                    # frame collections parsed.
                    # we no longer have the luxury of probability, just generate a second example
                    # for this frame collection and the next 4 future frame collections in the same
                    # category.

                    # generate another(second) example
                    _frames = np.array(frames, copy=True)
                    batch_examples.append(_frames)
                    batch_labels.append(label)
                    gen_num_examples += 1

                    # update the stat to reflect the current number of second example generated
                    cat_stats['5-6'][1] += 1

                elif cat_stats['5-6'][0] <= 4 and cat_stats['5-6'][1] < 5:
                    # we still have the luxury to flip a coin (random probability)
                    # to see if we should generate a second batch_examples using
                    # this frame collection, or push the responsiblity to a future frame collection
                    # of the same category (same label)

                    # flip a coin
                    outcome = np.random.binomial(1, 0.5)
                    # postive outcome, generate 'second' example
                    if outcome == 1:
                        # generate another(second) example
                        _frames = np.array(frames, copy=True)
                        batch_examples.append(_frames)
                        batch_labels.append(label)
                        gen_num_examples += 1

                        # update the stat to reflect the current number of second example generated
                        cat_stats['5-6'][1] += 1
                    else:
                        # negative outcome
                        # do nothing
                        pass
                        
                # sanity check
                if cat_stats['5-6'][0] == 10 and cat_stats['5-6'][1] != 5:
                    print('\n\nSomething went wrong some where')
                    print('Category: 3-4')
                    print('iteration: %d' % idx)
                    print('Exiting...\n\n')
                    sys.exit()

                if cat_stats['5-6'][0] == 10:
                    # reset stats for this category
                    cat_stats['5-6'][0] = 0
                    cat_stats['5-6'][1] = 0
            # end of if label >= 5  and label <= 6: 
     
            if int_label >= 7  and int_label <=  8: 
                # no scaling. just save example
                _frames = np.array(frames, copy=True)
                batch_examples.append(_frames)
                batch_labels.append(label)
                gen_num_examples += 1
            # end of if label >= 7 and label <= 8

            if int_label >= 9  and int_label <= 10: 
                # scale up, factor 1.5
                _frames = np.array(frames, copy=True)
                batch_examples.append(_frames)
                batch_labels.append(label)
                gen_num_examples += 1

                # sanity check
                if cat_stats['9-10'][0] > 10: 
                    print('\n\nSomething went wrong some where')
                    print('Category: 9-10')
                    print('iteration: %d' % idx)
                    print('Parsed example count > 10')
                    print('forgot to reset count somewhere')
                    print('Exiting...\n\n')
                    sys.exit()

                # update stat to denote that we have parsed through another frame collection
                # of this category (regardless of whether we generated a 2nd example from 
                # it or not)
                cat_stats['9-10'][0] += 1

                # use probablity to randomly determine if we are going to generate a second
                # example using this frame collection.
                # the '.5' in '1.5' scale factor means for a every 10 batch_examples with this same
                # label value, we generate a second example
                if cat_stats['9-10'][0] > 5 and cat_stats['9-10'][1] < 5:
                    # this means that if we have already parsed 5 frame collections
                    # and we have not yet generated a second example from the 5 previous 
                    # frame collections parsed.
                    # we no longer have the luxury of probability, just generate a second example
                    # for this frame collection and the next 4 future frame collections in the same
                    # category.

                    # generate another(second) example
                    _frames = np.array(frames, copy=True)
                    batch_examples.append(_frames)
                    batch_labels.append(label)
                    gen_num_examples += 1

                    # update the stat to reflect the current number of second example generated
                    cat_stats['9-10'][1] += 1

                elif cat_stats['9-10'][0] <= 4 and cat_stats['9-10'][1] < 5:
                    # we still have the luxury to flip a coin (random probability)
                    # to see if we should generate a second batch_examples using
                    # this frame collection, or push the responsiblity to a future frame collection
                    # of the same category (same label)

                    # flip a coin
                    outcome = np.random.binomial(1, 0.5)
                    # postive outcome, generate 'second' example
                    if outcome == 1:
                        # generate another(second) example
                        _frames = np.array(frames, copy=True)
                        batch_examples.append(_frames)
                        batch_labels.append(label)
                        gen_num_examples += 1

                        # update the stat to reflect the current number of second example generated
                        cat_stats['9-10'][1] += 1
                    else:
                        # negative outcome
                        # do nothing
                        pass

                # sanity check
                if cat_stats['9-10'][0] == 10 and cat_stats['9-10'][1] != 5:
                    print('\n\nSomething went wrong some where')
                    print('Category: 9-10')
                    print('iteration: %d' % idx)
                    print('Exiting...\n\n')
                    sys.exit()

                if cat_stats['9-10'][0] == 10:
                    # reset stats for this category
                    cat_stats['9-10'][0] = 0
                    cat_stats['9-10'][1] = 0
            # end of if label >= 9 and label <= 10

            if int_label >= 11 and int_label <= 12: 
                # scale up, factor 2.5

                # generate example 1
                _frames = np.array(frames, copy=True)
                batch_examples.append(_frames)
                batch_labels.append(label)
                gen_num_examples += 1

                # generate example 2
                _frames = np.array(frames, copy=True)
                batch_examples.append(_frames)
                batch_labels.append(label)
                gen_num_examples += 1

                # sanity check
                if cat_stats['11-12'][0] > 10: 
                    print('\n\nSomething went wrong some where')
                    print('Category: 11-12')
                    print('iteration: %d' % idx)
                    print('Parsed example count > 10')
                    print('forgot to reset count somewhere')
                    print('Exiting...\n\n')
                    sys.exit()

                # update stat to denote that we have parsed through another frame collection
                # of this category (regardless of the outcome of probability of generating 
                # a 3rd example from it
                cat_stats['11-12'][0] += 1

                # generate 3rd example based on probability

                # use probablity to randomly determine if we are going to generate a third
                # example using this frame collection.
                # the '.5' in '2.5' scale factor means for a every 10 batch_examples with this same
                # label value, we generate a third example
                if cat_stats['11-12'][0] > 5 and cat_stats['11-12'][1] < 5:
                    # this means that if we have already parsed 5 frame collections
                    # and we have not yet generated a second example from the 5 previous 
                    # frame collections parsed.
                    # we no longer have the luxury of probability, just generate a third example
                    # for this frame collection and the next 4 future frame collections in the same
                    # category.

                    # generate another(second) example
                    _frames = np.array(frames, copy=True)
                    batch_examples.append(_frames)
                    batch_labels.append(label)
                    gen_num_examples += 1

                    # update the stat to reflect the current number of third example generated
                    cat_stats['11-12'][1] += 1

                elif cat_stats['11-12'][0] <= 4 and cat_stats['11-12'][1] < 5:
                    # we still have the luxury to flip a coin (random probability)
                    # to see if we should generate a third batch_examples using
                    # this frame collection, or push the responsiblity to a future frame collection
                    # of the same category (same label)

                    # flip a coin
                    outcome = np.random.binomial(1, 0.5)
                    # postive outcome, generate 'third' example
                    if outcome == 1:
                        # generate another(third) example
                        _frames = np.array(frames, copy=True)
                        batch_examples.append(_frames)
                        batch_labels.append(label)
                        gen_num_examples += 1

                        # update the stat to reflect the current number of third example generated
                        cat_stats['11-12'][1] += 1
                    else:
                        # negative outcome
                        # do nothing
                        pass
                        
                # sanity check
                if cat_stats['11-12'][0] == 10 and cat_stats['11-12'][1] != 5:
                    print('\n\nSomething went wrong some where')
                    print('Category: 11-12')
                    print('iteration: %d' % idx)
                    print('Exiting...\n\n')
                    sys.exit()

                if cat_stats['11-12'][0] == 10:
                    # reset stats for this category
                    cat_stats['11-12'][0] = 0
                    cat_stats['11-12'][1] = 0
            # end of if label >= 11 and label <= 12

            if int_label >= 13 and int_label <= 14: 
                # scale up, factor 2.7

                # generate first example
                _frames = np.array(frames, copy=True)
                batch_examples.append(_frames)
                batch_labels.append(label)
                gen_num_examples += 1

                # generate second example
                _frames = np.array(frames, copy=True)
                batch_examples.append(_frames)
                batch_labels.append(label)
                gen_num_examples += 1

                # sanity check
                if cat_stats['13-14'][0] > 10: 
                    print('\n\nSomething went wrong some where')
                    print('Category: 13-14')
                    print('iteration: %d' % idx)
                    print('Parsed example count > 10')
                    print('forgot to reset count somewhere')
                    print('Exiting...\n\n')
                    sys.exit()

                # update stat to denote that we have parsed through another frame collection
                # of this category (regardless of the outcome of probability of generating
                # a 3rd example from it)
                cat_stats['13-14'][0] += 1

                # generate third example or not based on probability

                # use probablity to randomly determine if we are going to generate a third
                # example using this frame collection.
                # the '.7' in '2.7' scale factor means for a every 10 batch_examples with this same
                # label value, we generate 7 third example
                if cat_stats['13-14'][0] > 3 and cat_stats['13-14'][1] < 7:
                    # this means that if we have already parsed 3 frame collections
                    # and we have not yet generated a second example from the 3 previous 
                    # frame collections parsed.
                    # we no longer have the luxury of probability, just generate a third example
                    # for this frame collection and the next 6 future frame collections in the same
                    # category.

                    # generate another(third) example
                    _frames = np.array(frames, copy=True)
                    batch_examples.append(_frames)
                    batch_labels.append(label)
                    gen_num_examples += 1

                    # update the stat to reflect the current number of third example generated
                    cat_stats['13-14'][1] += 1

                elif cat_stats['13-14'][0] <= 2 and cat_stats['13-14'][1] < 7:
                    # we still have the luxury to flip a coin (random probability)
                    # to see if we should generate a second batch_examples using
                    # this frame collection, or push the responsiblity to a future frame collection
                    # of the same category (same label)

                    # flip a coin
                    outcome = np.random.binomial(1, 0.5)
                    # postive outcome, generate 'third' example
                    if outcome == 1:
                        # generate another(second) example
                        _frames = np.array(frames, copy=True)
                        batch_examples.append(_frames)
                        batch_labels.append(label)
                        gen_num_examples += 1

                        # update the stat to reflect the current number of third example generated
                        cat_stats['13-14'][1] += 1
                    else:
                        # negative outcome
                        # do nothing
                        pass
                        
                # sanity check
                if cat_stats['13-14'][0] == 10 and cat_stats['13-14'][1] != 7:
                    print('\n\nSomething went wrong some where')
                    print('Category: 13-14')
                    print('iteration: %d' % idx)
                    print('Exiting...\n\n')
                    sys.exit()

                if cat_stats['13-14'][0] == 10:
                    # reset stats for this category
                    cat_stats['13-14'][0] = 0
                    cat_stats['13-14'][1] = 0
            # end of if label >= 13 and label <= 14

            if int_label >= 15 and int_label <= 16: 
                # scale up, factor 2.2

                # generate first example
                _frames = np.array(frames, copy=True)
                batch_examples.append(_frames)
                batch_labels.append(label)
                gen_num_examples += 1

                # generate second example
                _frames = np.array(frames, copy=True)
                batch_examples.append(_frames)
                batch_labels.append(label)
                gen_num_examples += 1

                # sanity check
                if cat_stats['15-16'][0] > 10: 
                    print('\n\nSomething went wrong some where')
                    print('Category: 15-16')
                    print('iteration: %d' % idx)
                    print('Parsed example count > 10')
                    print('forgot to reset count somewhere')
                    print('Exiting...\n\n')
                    sys.exit()


                # update stat to denote that we have parsed through another frame collection
                # of this category (regardless of the outcome of probability of generating 
                # a 3rd example from it)
                cat_stats['15-16'][0] += 1

                # generate third example or not based on probability

                # use probablity to randomly determine if we are going to generate a third
                # example using this frame collection.
                # the '.2' in '2.2' scale factor means for a every 10 batch_examples with this same
                # label value, we generate two third example
                if cat_stats['15-16'][0] > 8 and cat_stats['15-16'][1] < 2:
                    # this means that if we have already parsed 2 frame collections
                    # and we have not yet generated a third example from the 2 previous 
                    # frame collections parsed.
                    # we no longer have the luxury of probability, just generate a third example
                    # for this frame collection and the next 7 future frame collections in the same
                    # category.

                    # generate another(third) example
                    _frames = np.array(frames, copy=True)
                    batch_examples.append(_frames)
                    batch_labels.append(label)
                    gen_num_examples += 1

                    # update the stat to reflect the current number of third example generated
                    cat_stats['15-16'][1] += 1

                elif cat_stats['15-16'][0] <= 7 and cat_stats['15-16'][1] < 2:
                    # we still have the luxury to flip a coin (random probability)
                    # to see if we should generate a third batch_examples using
                    # this frame collection, or push the responsiblity to a future frame collection
                    # of the same category (same label)

                    # flip a coin
                    outcome = np.random.binomial(1, 0.5)
                    # postive outcome, generate 'third' example
                    if outcome == 1:
                        # generate another(third) example
                        _frames = np.array(frames, copy=True)
                        batch_examples.append(_frames)
                        batch_labels.append(label)
                        gen_num_examples += 1

                        # update the stat to reflect the current number of third example generated
                        cat_stats['15-16'][1] += 1
                    else:
                        # negative outcome
                        # do nothing
                        pass


                # sanity check
                if cat_stats['15-16'][0] == 10 and cat_stats['15-16'][1] != 2:
                    print('\n\nSomething went wrong some where')
                    print('Category: 15-16')
                    print('iteration: %d' % idx)
                    print('Exiting...\n\n')
                    sys.exit()

                if cat_stats['15-16'][0] == 10:
                    # reset stats for this category
                    cat_stats['15-16'][0] = 0
                    cat_stats['15-16'][1] = 0
            # end of if label >= 15 and label <= 16

            if int_label >= 17 and int_label <= 18: 
                # scale up, factor 1.9

                # generate first example
                _frames = np.array(frames, copy=True)
                batch_examples.append(_frames)
                batch_labels.append(label)
                gen_num_examples += 1

                # sanity check
                if cat_stats['17-18'][0] > 10: 
                    print('\n\nSomething went wrong some where')
                    print('Category: 17-18')
                    print('iteration: %d' % idx)
                    print('Parsed example count > 10')
                    print('forgot to reset count somewhere')
                    print('Exiting...\n\n')
                    sys.exit()

                # update stat to denote that we have parsed through another frame collection
                # of this category (regardless of the outcome of probability of generating 
                # a 3rd example from it)
                cat_stats['17-18'][0] += 1

                # generate second example or not based on probability

                # use probablity to randomly determine if we are going to generate a second
                # example using this frame collection.
                # the '.9' in '1.9' scale factor means for a every 10 batch_examples with this same
                # label value, we generate a second example
                if cat_stats['17-18'][0] > 1 and cat_stats['17-18'][1] < 9:
                    # this means that if we have already parsed 3 frame collections
                    # and we have not yet generated a second example from the 3 previous 
                    # frame collections parsed.
                    # we no longer have the luxury of probability, just generate a second example
                    # for this frame collection and the next 6 future frame collections in the same
                    # category.

                    # generate another(second) example
                    _frames = np.array(frames, copy=True)
                    batch_examples.append(_frames)
                    batch_labels.append(label)
                    gen_num_examples += 1

                    # update the stat to reflect the current number of second example generated
                    cat_stats['17-18'][1] += 1

                elif cat_stats['17-18'][0] <= 1 and cat_stats['17-18'][1] < 9:
                    # we still have the luxury to flip a coin (random probability)
                    # to see if we should generate a second batch_examples using
                    # this frame collection, or push the responsiblity to a future frame collection
                    # of the same category (same label)

                    # flip a coin
                    outcome = np.random.binomial(1, 0.5)
                    # postive outcome, generate 'second' example
                    if outcome == 1:
                        # generate another(second) example
                        _frames = np.array(frames, copy=True)
                        batch_examples.append(_frames)
                        batch_labels.append(label)
                        gen_num_examples += 1

                        # update the stat to reflect the current number of second example generated
                        cat_stats['17-18'][1] += 1
                    else:
                        # negative outcome
                        # do nothing
                        pass


                # sanity check
                if cat_stats['17-18'][0] == 10 and cat_stats['17-18'][1] != 9:
                    print('\n\nSomething went wrong some where')
                    print('Category: 17-18')
                    print('iteration: %d' % idx)
                    print('Exiting...\n\n')
                    sys.exit()

                if cat_stats['17-18'][0] == 10:
                    # reset stats for this category
                    cat_stats['17-18'][0] = 0
                    cat_stats['17-18'][1] = 0
            # end of if label >= 17 and label <= 18

            if int_label >= 19 and int_label <= 20: 
                # scale up, factor 2.2

                # generate first example
                _frames = np.array(frames, copy=True)
                batch_examples.append(_frames)
                batch_labels.append(label)
                gen_num_examples += 1

                # generate second example
                _frames = np.array(frames, copy=True)
                batch_examples.append(_frames)
                batch_labels.append(label)
                gen_num_examples += 1

                # sanity check
                if cat_stats['19-20'][0] > 10: 
                    print('\n\nSomething went wrong some where')
                    print('Category: 19-20')
                    print('iteration: %d' % idx)
                    print('Parsed example count > 10')
                    print('forgot to reset count somewhere')
                    print('Exiting...\n\n')
                    sys.exit()

                # update stat to denote that we have parsed through another frame collection
                # of this category (regardless of the outcome of probability of generating a 3rd example from it
                cat_stats['19-20'][0] += 1

                # generate third example or not based on probability

                # use probablity to randomly determine if we are going to generate a third
                # example using this frame collection.
                # the '.2' in '2.2' scale factor means for a every 10 batch_examples with this same
                # label value, we generate two third example
                if cat_stats['19-20'][0] > 8 and cat_stats['19-20'][1] < 2:
                    # this means that if we have already parsed 8 frame collections
                    # and we have not yet generated a third example from the 8 previous 
                    # frame collections parsed.
                    # we no longer have the luxury of probability, just generate a third example
                    # for this frame collection and the next 2 future frame collections in the same
                    # category.

                    # generate another(third) example
                    _frames = np.array(frames, copy=True)
                    batch_examples.append(_frames)
                    batch_labels.append(label)
                    gen_num_examples += 1

                    # update the stat to reflect the current number of third example generated
                    cat_stats['19-20'][1] += 1

                elif cat_stats['19-20'][0] <= 7 and cat_stats['19-20'][1] < 2:
                    # we still have the luxury to flip a coin (random probability)
                    # to see if we should generate a second batch_examples using
                    # this frame collection, or push the responsiblity to a future frame collection
                    # of the same category (same label)

                    # flip a coin
                    outcome = np.random.binomial(1, 0.5)
                    # postive outcome, generate 'third' example
                    if outcome == 1:
                        # generate another(second) example
                        _frames = np.array(frames, copy=True)
                        batch_examples.append(_frames)
                        batch_labels.append(label)
                        gen_num_examples += 1

                        # update the stat to reflect the current number of third example generated
                        cat_stats['19-20'][1] += 1
                    else:
                        # negative outcome
                        # do nothing
                        pass

                # sanity check
                if cat_stats['19-20'][0] == 10 and cat_stats['19-20'][1] != 2:
                    print('\n\nSomething went wrong some where')
                    print('Category: 19-20')
                    print('iteration: %d' % idx)
                    print('Exiting...\n\n')
                    sys.exit()

                if cat_stats['19-20'][0] == 10:
                    # reset stats for this category
                    cat_stats['19-20'][0] = 0
                    cat_stats['19-20'][1] = 0
            # end of if label >= 19 and label <= 20

            if int_label >= 21 and int_label <= 22: 
                # scale down, factor 0.54
                # i.e for every 100 frame collections of this category,
                # only use 54 to generate batch_examples, discard the other 36

                # sanity check
                if cat_stats['21-22'][0] > 100: 
                    print('\n\nSomething went wrong some where')
                    print('Category: 21-22')
                    print('iteration: %d' % idx)
                    print('Parsed example count > 100')
                    print('forgot to reset count somewhere')
                    print('Exiting...\n\n')
                    sys.exit()


                # update stat to denote that we have parsed through another frame collection
                # of this category (regardless of the outcome of probability of generating a 3rd example from it
                cat_stats['21-22'][0] += 1

                if cat_stats['21-22'][0] > 36 and cat_stats['21-22'][1] < 54:
                    # we no longer have the luxury of probability, just generate a second example
                    # for this frame collection and the next 6 future frame collections in the same
                    # category.

                    # generate example
                    _frames = np.array(frames, copy=True)
                    batch_examples.append(_frames)
                    batch_labels.append(label)
                    gen_num_examples += 1

                    cat_stats['21-22'][1] += 1

                elif cat_stats['21-22'][0] <= 35 and cat_stats['21-22'][1] < 54:
                    # we still have the luxury to flip a coin (random probability)
                    # to see if we should generate a second batch_examples using
                    # this frame collection, or push the responsiblity to a future frame collection
                    # of the same category (same label)

                    # flip a coin
                    outcome = np.random.binomial(1, 0.5)
                    # postive outcome, generate example
                    if outcome == 1:
                        # generate example
                        _frames = np.array(frames, copy=True)
                        batch_examples.append(_frames)
                        batch_labels.append(label)
                        gen_num_examples += 1

                        cat_stats['21-22'][1] += 1
                    else:
                        # negative outcome
                        # do nothing
                        pass

                # sanity check
                if cat_stats['21-22'][0] == 100 and cat_stats['21-22'][1] != 54:
                    print('\n\nSomething went wrong some where')
                    print('Category: 21-22')
                    print('iteration: %d' % idx)
                    print('Exiting...\n\n')
                    sys.exit()

                if cat_stats['21-22'][0] == 100:
                    # reset stats for this category
                    cat_stats['21-22'][0] = 0
                    cat_stats['21-22'][1] = 0

            # end of if label >= 21 and label <= 22

            if int_label >= 23 and int_label <= 24: 
                # no scaling. just save example
                _frames = np.array(frames, copy=True)
                batch_examples.append(_frames)
                batch_labels.append(label)
                gen_num_examples += 1
            # end of if label >= 23 and label <= 24

            if int_label >= 25 and int_label <= 26: 
                # scale up, factor 2.3

                # generate first example
                _frames = np.array(frames, copy=True)
                batch_examples.append(_frames)
                batch_labels.append(label)
                gen_num_examples += 1

                # generate second example
                _frames = np.array(frames, copy=True)
                batch_examples.append(_frames)
                batch_labels.append(label)
                gen_num_examples += 1

                # sanity check
                if cat_stats['25-26'][0] > 10: 
                    print('\n\nSomething went wrong some where')
                    print('Category: 25-26')
                    print('iteration: %d' % idx)
                    print('Parsed example count > 10')
                    print('forgot to reset count somewhere')
                    print('Exiting...\n\n')
                    sys.exit()

                # update stat to denote that we have parsed through another frame collection
                # of this category (regardless of the outcome of probability of generating a 3rd example from it
                cat_stats['25-26'][0] += 1

                # generate third example or not based on probability
                
                # use probablity to randomly determine if we are going to generate a third
                # example using this example.
                # the '.3' in '2.3' scale factor means for a every 10 batch_examples with this same
                # label value, we generate a third example
                if cat_stats['25-26'][0] > 7 and cat_stats['25-26'][1] < 3:
                    # this means that if we have already parsed 7 frame collections
                    # and we have not yet generated a third example from the 7 previous frame collections
                    # parsed.
                    # we no longer have the luxury of probability, just generate a third example
                    # for this frame collection and the next 2 future frame collections in the same
                    # category.

                    # generate another(third) example
                    _frames = np.array(frames, copy=True)
                    batch_examples.append(_frames)
                    batch_labels.append(label)
                    gen_num_examples += 1

                    # update stat to denote number of 3rd example generated
                    cat_stats['25-26'][1] += 1

                elif cat_stats['25-26'][0] <= 6 and cat_stats['25-26'][1] < 3:
                    # we still have the luxury to flip a coin (random probability)
                    # to see if we should generate a third batch_examples using
                    # this frame collection, or push the responsiblity to a future frame collection
                    # of the same category (same label)

                    # flip a coin
                    outcome = np.random.binomial(1, 0.5)
                    # postive outcome, generate 'third' example
                    if outcome == 1:
                        # generate another(third) example
                        _frames = np.array(frames, copy=True)
                        batch_examples.append(_frames)
                        batch_labels.append(label)
                        gen_num_examples += 1

                        # update stat to denote the current number of 3rd example generated
                        cat_stats['25-26'][1] += 1 
                    else:
                        # negative outcome
                        # do nothing
                        pass
                    
                # sanity check
                if cat_stats['25-26'][0] == 10 and cat_stats['25-26'][1] != 3:
                    print('\n\nSomething went wrong some where')
                    print('Category: 25-26')
                    print('iteration: %d' % idx)
                    print('Exiting...\n\n')
                    sys.exit()

                if cat_stats['25-26'][0] == 10:
                    # reset stats for this category
                    cat_stats['25-26'][0] = 0
                    cat_stats['25-26'][1] = 0
            # end of if label >= 25 and label <= 26

            if int_label >= 27 and int_label <= 28: 
                # scale up, factor 48

                # generate 48 example for this category
                for i in np.arange(48):
                    _frames = np.array(frames, copy=True)
                    batch_examples.append(_frames)
                    batch_labels.append(label)
                    gen_num_examples += 1
            # end of if label >= 27 and label <= 28

            if len(batch_examples) >= BATCH_SIZE: # one batch is completed
                # save batch batch_examples to disk
                np.savez(output_dir + 'batch_{0}.npz'.format(batch_num+1), 
                   examples=np.asarray(batch_examples[ : BATCH_SIZE]), 
                   labels=np.expand_dims(np.asarray(batch_labels[ : BATCH_SIZE], dtype=np.float32), 1))
                # reset batch_examples and label list
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
        sys.stdout.write('\rNum of batch_examples generated: %d of %d' % gen_num_examples)

    # close video file
    cap.release()

    print('\ncompleted...')
    print('number of batch_examples generated: %d' % gen_num_examples)
    print('total number of batches saved to disk: %d' % batch_num)

    return

if __name__ == '__main__':
    print('\ngenerating training example clips from training video split\n')
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

