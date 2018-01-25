import argparse
from keras.optimizers import Adam
from utils import *

NUM_FRAMES_PER_EXAMPLE = 40
FRAME_HEIGHT = 64
FRAME_WIDTH = 64
NUM_RGB_CHANNELS = 3
NUM_FLOW_CHANNELS = 2

# configs
batch_size = 16 

# config related to train/val split
TOTAL_NUM_TRAIN_SPLIT_EXAMPLES = get_total_num_examples('data/rgb_train_val_split/train_data/') 
TOTAL_NUM_VAL_SPLIT_EXAMPLES = get_total_num_examples('data/rgb_train_val_split/validation_data/') 

def main(args):
    # train and validation generator 
    train_data_generator = video_data_generator('data/rgb_train_val_split/train_data/',
                                                batch_size=batch_size,
                                                shuffle=False,
                                                center_crop=True,
                                                two_channels_frame=False)

    val_data_generator = video_data_generator('data/rgb_train_val_split/validation_data/',
                                              batch_size=batch_size,
                                              shuffle=False,
                                              center_crop=True,
                                              two_channels_frame=False)
    

    # Build model
    model = build_model(NUM_FRAMES_PER_EXAMPLE, FRAME_HEIGHT, FRAME_WIDTH, NUM_RGB_CHANNELS, 
                        base_model_weights=None, freeze_base=False)

    # load saved model weights
    model.load_weights(args.weights_path)

    # compile model
    optimizer = Adam(lr=0.0001)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    print(model.summary())

    # get the number of batches in training and validation data based on `batch_size`
    train_steps = TOTAL_NUM_TRAIN_SPLIT_EXAMPLES // batch_size
    val_steps = TOTAL_NUM_VAL_SPLIT_EXAMPLES // batch_size

    # evaluate model
    if args.data_type == 'both' or args.data_type == 'validation':
        print('\ngenerating predictions for validation data')
        cache_preds = []
        cache_y_vals = []
        step_count = 0
        while True:
            X_val, y_val = next(val_data_generator)
            preds = model.predict(X_val)
            
            cache_y_vals += y_val.reshape((y_val.shape[0], )).tolist()
            cache_preds +=  preds.reshape((preds.shape[0], )).tolist()

            step_count += 1

            if step_count >= val_steps:
                break

        _predictions = np.array(cache_preds)
        _ground_truth = np.array(cache_y_vals)
        loss = np.sum(np.square(np.subtract(_predictions, _ground_truth))) / _predictions.shape[0]
        print('loss: %f' % loss) # mean squared error

        with open('rgb_val_predictions.txt', 'w') as f:
            for i in np.arange(len(cache_preds)):
                f.write('prediction: %f, actual: %f\n' % (cache_preds[i], cache_y_vals[i]))

    if args.data_type == 'both' or args.data_type == 'train':
        print('\ngenerating predictions for training data')
        cache_preds = []
        cache_y_train = []
        step_count = 0
        while True:
            X_train, y_train = next(train_data_generator)
            preds = model.predict(X_train)
            
            cache_y_train += y_train.reshape((y_train.shape[0], )).tolist()
            cache_preds +=  preds.reshape((preds.shape[0], )).tolist()

            step_count += 1

            if step_count >= train_steps:
                break

        _predictions = np.array(cache_preds)
        _ground_truth = np.array(cache_y_train)
        loss = np.sum(np.square(np.subtract(_predictions, _ground_truth))) / _predictions.shape[0]
        print('loss: %f' % loss) # mean squared error

        with open('rgb_train_predictions.txt', 'w') as f:
            for i in np.arange(len(cache_preds)):
                f.write('prediction: %f, actual: %f\n' % (cache_preds[i], cache_y_train[i]))

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('weights_path',
                        help='Path to the trained weights of the model',
                        type=str)
    parser.add_argument('-t', '--data-type',
                        help='Evaluate on training data or validation data', 
                        type=str,
                        choices=['train', 'validation', 'both'],
                        default='validation')
    main(parser.parse_args())
