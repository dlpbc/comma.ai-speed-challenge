import argparse
import time

from keras.optimizers import Adam

from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import CSVLogger
from keras.callbacks import EarlyStopping

from utils import *

# configs
NUM_FRAMES_PER_EXAMPLE = 40
FRAME_HEIGHT = 64
FRAME_WIDTH = 64
NUM_RGB_CHANNELS = 3
NUM_FLOW_CHANNELS = 2

# hyper parameters 
epochs = 20
batch_size = 16 
lr = 0.0001

TOTAL_NUM_TRAIN_SPLIT_EXAMPLES = get_total_num_examples('data/flow_train_val_split/train_data/') 
TOTAL_NUM_VAL_SPLIT_EXAMPLES = get_total_num_examples('data/flow_train_val_split/validation_data/') 

checkpoint_path = 'flow_saved_weights/weights.{epoch:02d}-{val_loss:.2f}.hdf5'

def main(args):
    # train and validation generator 
    train_data_generator = video_data_generator('data/flow_train_val_split/train_data/',
                                                batch_size=batch_size,
                                                shuffle=True,
                                                center_crop=True,
                                                two_channels_frame=True)

    val_data_generator = video_data_generator('data/flow_train_val_split/validation_data/',
                                              batch_size=batch_size,
                                              shuffle=False,
                                              center_crop=True,
                                              two_channels_frame=True)
    

    # Build model
    base_model_weights = None if args.weights else 'flow_imagenet_and_kinetics'

    model = build_model(NUM_FRAMES_PER_EXAMPLE, FRAME_HEIGHT, FRAME_WIDTH, NUM_FLOW_CHANNELS, 
                        base_model_weights, freeze_base=False)

    if args.weights:
        # load saved model weights
        model.load_weights(args.weights)

    # compile model
    optimizer = Adam(lr=lr)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    print(model.summary())

    # get the number of batches in training and validation data based on `batch_size`
    train_steps = TOTAL_NUM_TRAIN_SPLIT_EXAMPLES // batch_size
    val_steps = TOTAL_NUM_VAL_SPLIT_EXAMPLES // batch_size

    # define callbacks: learning rate annealer and save checkpoints
    reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                  factor=0.1, 
                                  patience=3,
                                  verbose=1,
                                  min_lr=1e-6,
                                  cooldown=2)
    save_checkpoints = ModelCheckpoint(filepath=checkpoint_path,
                                       monitor='val_loss',
                                       verbose=1,
                                       save_best_only=True,
                                       save_weights_only=True)
    early_stopper = EarlyStopping(monitor='val_loss', patience=3)
    csv_logger = CSVLogger('flow_train_log.txt', separator=',', append=True)

    # fit model
    model.fit_generator(train_data_generator, 
                        steps_per_epoch=train_steps,
                        epochs=epochs,
                        validation_data=val_data_generator,
                        validation_steps=val_steps,
                        callbacks=[reduce_lr, save_checkpoints, early_stopper, csv_logger])

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--weights', 
                        help='path to weights to load',
                        type=str,
                        default=None)
    main(parser.parse_args())
