'''
The script takes 2 npz batch files (containing example clips) and shuffles them.

The reason for this is because a single batch file can contain a large number of
examples within a speed (label) range category. A batch might contain more of 11-15 speed
range example clips than any other speed range and this affects training. The entire
dataset needs to be properly shuffled.

This issue stems from the nature of video frames speed label, since you can have several
number of consecutive frames within thesame speed range.

'''
import os
import argparse
import numpy as np

'''
this takes 2 npz files (where number of examples in each is not thesame), concatenates their examples, 
shuffles them, and then split into 2 parts with the size of the original files

Args:
    file1_path: path to the first npz file
    file2_path: path to the second npz file
    save_dir: directory to save the new npz files

'''
def two_split_mixer(file1_path, file2_path, save_dir=None):
    # load npz files
    file1 = np.load(file1_path)
    file2 = np.load(file2_path)

    X_data1 = file1['examples']
    y_data1 = file1['labels']
    X_data2 = file2['examples']
    y_data2 = file2['labels']

    # initial data shuffle before slicing
    # data1
    indexes = np.arange(X_data1.shape[0])
    np.random.shuffle(indexes)
    X_data1 = X_data1[indexes]
    y_data1 = y_data1[indexes]
    # data2
    indexes = np.arange(X_data2.shape[0])
    np.random.shuffle(indexes)
    X_data2 = X_data2[indexes]
    y_data2 = y_data2[indexes]

    # concatenate
    X_data = np.concatenate([X_data1, X_data2], axis=0)
    y_data = np.concatenate([y_data1, y_data2], axis=0)

    # shuffle
    indexes = np.arange(X_data.shape[0])
    np.random.shuffle(indexes)
    X_data = X_data[indexes]
    y_data = y_data[indexes]

    # split
    file1_num_examples = X_data1.shape[0]
    new_X_data1 = X_data[ : file1_num_examples]
    new_y_data1 = y_data[ : file1_num_examples]

    new_X_data2 = X_data[file1_num_examples : ]
    new_y_data2 = y_data[file1_num_examples : ]
    
    save_file1_path = file1_path.split('/')[-1]
    save_file2_path = file2_path.split('/')[-1]
    if save_dir is not None:
        save_file1_path = '{0}/{1}'.format(save_dir, save_file1_path)
        save_file2_path = '{0}/{1}'.format(save_dir, save_file2_path)

    np.savez(save_file1_path, examples=new_X_data1, labels=new_y_data1)
    np.savez(save_file2_path, examples=new_X_data2, labels=new_y_data2)

    return


def main(args):
    if args.save_dir is not None:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

    two_split_mixer(args.npz_file1_path, args.npz_file2_path, args.save_dir)
    
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('npz_file1_path',
                        help='Path to the first npz file',
                        type=str)
    parser.add_argument('npz_file2_path',
                        help='Path to the second npz file',
                        type=str)
    parser.add_argument('--save-dir',
                        help='Directory to save generated files',
                        type=str)

    main(parser.parse_args())
