import sys
import argparse
import numpy as np
from utils import build_model
from test_utils import generate_test_data

NUM_FRAMES_PER_EXAMPLE = 40
FRAME_HEIGHT = 64
FRAME_WIDTH = 64
NUM_RGB_CHANNELS = 3
NUM_FLOW_CHANNELS = 2

RGB_TOTAL_NUM_TEST_VIDEO_FRAMES = 10798
RGB_NUM_TEST_EXAMPLES = RGB_TOTAL_NUM_TEST_VIDEO_FRAMES - NUM_FRAMES_PER_EXAMPLE + 1

FLOW_TOTAL_NUM_TEST_VIDEO_FRAMES = 10797
FLOW_NUM_TEST_EXAMPLES = FLOW_TOTAL_NUM_TEST_VIDEO_FRAMES - NUM_FRAMES_PER_EXAMPLE + 1

batch_size = 16
rgb_video_path = 'data/test.mp4'
flow_video_path = 'data/flow_test.mp4'
prediction_save_path = 'test_predictions.txt'

def main(args):
    if args.eval_type == 'joint' and (args.rgb_weights_path is None or args.flow_weights_path is None):
        print('ERROR: when using `joint` model evaluation type, '
              'you must specify path to RGB and Optical flow weights respectively.')
        print('Exiting...')
        return
    elif args.eval_type == 'rgb' and args.rgb_weights_path is None:
        print('ERROR: when using only `rgb` model evaluation type, '
              'you must specify path to RGB weights.')
        print('Exiting...')
        return
    elif args.eval_type == 'flow' and args.flow_weights_path is None:
        print('ERROR: when using only `flow` model evaluation type, '
              'you must specify path to Optical Flow weights.')
        print('Exiting...')

    # define rgb model, load trained weights and evaluate test samples
    if args.eval_type in ['joint', 'rgb']:
        print('\nprocessing RGB video')
        print('path: %s' % rgb_video_path)
        rgb_model = build_model(NUM_FRAMES_PER_EXAMPLE, FRAME_HEIGHT, FRAME_WIDTH, 
                                NUM_RGB_CHANNELS, base_model_weights=None, freeze_base=False)
        rgb_model.load_weights(args.rgb_weights_path)
        # define data generator
        rgb_generator = generate_test_data(rgb_video_path, NUM_FRAMES_PER_EXAMPLE, 
                                           batch_size, two_channels_frame=False)
        # evaluate
        if RGB_NUM_TEST_EXAMPLES % batch_size == 0:
            num_batches = RGB_NUM_TEST_EXAMPLES // batch_size
        else:
            num_batches = (RGB_NUM_TEST_EXAMPLES // batch_size) + 1
        rgb_predictions = []
        step = 0
        while True:
            X_test = next(rgb_generator)
            preds = rgb_model.predict(X_test)
            rgb_predictions += preds.reshape((preds.shape[0],)).tolist()
            step += 1
            sys.stdout.write('num of batches processed: %d of %d\r' % (step, num_batches))
            if step >= num_batches:
                break
        rgb_predictions = np.asarray(rgb_predictions, dtype=np.float32)


    # define optical flow model, load trained weights and evaluate test samples
    if args.eval_type in ['joint', 'flow']:
        print('\n\nprocessing Optical Flow video')
        print('path: %s' % flow_video_path)
        flow_model = build_model(NUM_FRAMES_PER_EXAMPLE, FRAME_HEIGHT, FRAME_WIDTH, 
                                 NUM_FLOW_CHANNELS, base_model_weights=None, freeze_base=False)
        flow_model.load_weights(args.flow_weights_path)
        # define data generator
        flow_generator = generate_test_data(flow_video_path, NUM_FRAMES_PER_EXAMPLE, 
                                            batch_size, two_channels_frame=True)
        # evaluate
        if FLOW_NUM_TEST_EXAMPLES % batch_size == 0:
            num_batches = FLOW_NUM_TEST_EXAMPLES // batch_size
        else:
            num_batches = (FLOW_NUM_TEST_EXAMPLES // batch_size) + 1
        flow_predictions = []
        step = 0
        while True:
            X_test = next(flow_generator)
            preds = flow_model.predict(X_test)
            flow_predictions += preds.reshape((preds.shape[0],)).tolist()
            step += 1
            sys.stdout.write('num of batches processed: %d of %d\r' % (step, num_batches))
            if step >= num_batches:
                break
        flow_predictions = np.asarray(flow_predictions, dtype=np.float32)

    if args.eval_type == 'rgb':
        model_predictions = rgb_predictions
    elif args.eval_type == 'flow':
        # flow examples are one less than rgb examples
        # therefore, duplicate the first flow prediction
        # so that number of predictions is same as rgb examples
        model_predictions = np.zeros_like(flow_predictions.shape[0] + 1, dtype=np.float32)
        model_predictions[1:] = flow_predictions[:]
        model_predictions[0] = flow_predictions[0]
    else:
        model_predictions = np.zeros_like(rgb_predictions, dtype=np.float32)
        # combine rgb prediction and flow prediction
        model_predictions[1:] = (rgb_predictions[1:] + flow_predictions[:]) / 2.0
        model_predictions[0] = (rgb_predictions[0] + flow_predictions[0]) / 2.0
      
    # save predictions
    f = open(prediction_save_path, 'w')
    # use the first prediction in `model_predictions`
    # as the predictions for the first 39 frames in test video
    # since we started generating examples from the 40th frame onwards
    for _ in range(39):
        f.write('{0}\n'.format(model_predictions[0]))

    # save the predictions that starts from the 40th frame onwards
    for prediction in model_predictions:
        f.write('{0}\n'.format(prediction))

    f.close()

    print('\nprediction generation completed')
    print('prediction output file: %s' % prediction_save_path)


    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('eval_type',
                        help='strategy to evaluate test data. '  
                             'if `joint`, you must provide weights path for both rgb and flow. '
                             'if `rgb`, you must provide weights path for rgb. '
                             'if `flow`, you must provide weights path for flow. ',
                        type=str,
                        choices=['joint', 'rgb', 'flow'])

    parser.add_argument('-r', '--rgb-weights-path',
                        help='path to weights for RGB model',
                        type=str)

    parser.add_argument('-f', '--flow-weights-path',
                        help='path to weights for flow model',
                        type=str)

    main(parser.parse_args())

