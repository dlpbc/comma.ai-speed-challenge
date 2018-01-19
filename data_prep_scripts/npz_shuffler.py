'''
The script takes 2 or 3 npz batch files (containing example clips) and shuffles them.

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
this takes 2 npz files containing examples, split each of them into 2 parts, 
which is used to produce 2 new files by mixing. The idea is described below

The first new file is produced by combining:
    first slice of the first file + second slice of the second file

The second new file is produced by combinining:
    first slice of the second file + second slice of the first file


Example: Given the files slices below:

    first file:  | 1.1 | 1.2 |
    second file: | 2.1 | 2.2 |

we can produce three new files by combining slices (of current files) as shown below:

    first new file:  | 1.1 | 2.2 |
    second new file: | 2.1 | 1.2 |

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


    # sanity check
    if not (X_data1.shape[0] == X_data2.shape[0]): 
        print('number of examples in the two files should be thesame')
        print('please correct your npz files and try again')
        return


    half_point_idx = X_data1.shape[0] // 2

    # data1 X, y slices
    first_slice_X_data1    = X_data1[ : half_point_idx]
    second_slice_X_data1   = X_data1[half_point_idx : ]

    first_slice_y_data1    = y_data1[ : half_point_idx]
    second_slice_y_data1   = y_data1[half_point_idx : ]

    # data2 X, y slices
    first_slice_X_data2    = X_data2[ : half_point_idx]
    second_slice_X_data2   = X_data2[half_point_idx : ]

    first_slice_y_data2    = y_data2[ : half_point_idx]
    second_slice_y_data2   = y_data2[half_point_idx : ]


    new_X_data1 = np.concatenate([first_slice_X_data1, 
                                  second_slice_X_data2],
                                  axis=0)
    new_X_data2 = np.concatenate([first_slice_X_data2, 
                                  second_slice_X_data1],
                                  axis=0)

    new_y_data1 = np.concatenate([first_slice_y_data1, 
                                  second_slice_y_data2],
                                  axis=0)
    new_y_data2 = np.concatenate([first_slice_y_data2, 
                                  second_slice_y_data1],
                                  axis=0)

    # shuffle data
    # data1
    indexes = np.arange(new_X_data1.shape[0])
    np.random.shuffle(indexes)
    new_X_data1 = new_X_data1[indexes]
    new_y_data1 = new_y_data1[indexes]

    # data2
    indexes = np.arange(new_X_data2.shape[0])
    np.random.shuffle(indexes)
    new_X_data2 = new_X_data2[indexes]
    new_y_data2 = new_y_data2[indexes]

    save_file1_path = file1_path.split('/')[-1]
    save_file2_path = file2_path.split('/')[-1]
    if save_dir is not None:
        save_file1_path = '{0}/{1}'.format(save_dir, save_file1_path)
        save_file2_path = '{0}/{1}'.format(save_dir, save_file2_path)

    np.savez(save_file1_path, examples=new_X_data1, labels=new_y_data1)
    np.savez(save_file2_path, examples=new_X_data2, labels=new_y_data2)

    return


'''
this takes 3 npz files containing examples, split each of them into 3 parts, 
which is used to produce 3 new files by mixing. The idea is described below

The first new file is produced by combining:
    first slice of the first file + second slice of the second file + third slice of third file

The second new file is produced by combinining:
    first slice of the second file + second slice of the third file + third slice of the first file

The third new file is produced by combining:
    first slice of the third file + second lsice of the first file + third slice of the second file

Example: Given the files slices below:

    first file:  | 1.1 | 1.2 | 1.3 |
    second file: | 2.1 | 2.2 | 2.3 |
    third file:  | 3.1 | 3.2 | 3.3 |

we can produce three new files by combining slices (of current files) as shown below:

    first new file:  | 1.1 | 2.2 | 3.3 |
    second new file: | 2.1 | 3.2 | 1.3 |
    third new file:  | 3.1 | 1.2 | 2.3 |

Args:
    file1_path: path to the first npz file
    file2_path: path to the second npz file
    file3_path: path to the third npz file
    save_dir: directory to save the new npz files

'''
def three_split_mixer(file1_path, file2_path, file3_path, save_dir=None):
    # load npz files
    file1 = np.load(file1_path)
    file2 = np.load(file2_path)
    file3 = np.load(file3_path)

    X_data1 = file1['examples']
    y_data1 = file1['labels']
    X_data2 = file2['examples']
    y_data2 = file2['labels']
    X_data3 = file3['examples']
    y_data3 = file3['labels']

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
    # data3
    indexes = np.arange(X_data3.shape[0])
    np.random.shuffle(indexes)
    X_data3 = X_data3[indexes]
    y_data3 = y_data3[indexes]


    # sanity check
    if not (X_data1.shape[0] == X_data2.shape[0] == X_data3.shape[0]): 
        print('number of examples in the three files should be thesame')
        print('please correct your npz files and try again')
        return

    one_third_idx = X_data1.shape[0] // 3
    two_third_idx = (one_third_idx + one_third_idx)  

    # data1 X, y slices
    first_slice_X_data1    = X_data1[ : one_third_idx]
    second_slice_X_data1   = X_data1[one_third_idx : two_third_idx]
    third_slice_X_data1    = X_data1[two_third_idx : ]

    first_slice_y_data1    = y_data1[ : one_third_idx]
    second_slice_y_data1   = y_data1[one_third_idx : two_third_idx]
    third_slice_y_data1    = y_data1[two_third_idx : ]

    # data2 X, y slices
    first_slice_X_data2    = X_data2[ : one_third_idx]
    second_slice_X_data2   = X_data2[one_third_idx : two_third_idx]
    third_slice_X_data2    = X_data2[two_third_idx : ]

    first_slice_y_data2    = y_data2[ : one_third_idx]
    second_slice_y_data2   = y_data2[one_third_idx : two_third_idx]
    third_slice_y_data2    = y_data2[two_third_idx : ]

    # data3 X, y slices
    first_slice_X_data3    = X_data3[ : one_third_idx]
    second_slice_X_data3   = X_data3[one_third_idx : two_third_idx]
    third_slice_X_data3    = X_data3[two_third_idx : ]

    first_slice_y_data3    = y_data3[ : one_third_idx]
    second_slice_y_data3   = y_data3[one_third_idx : two_third_idx]
    third_slice_y_data3    = y_data3[two_third_idx : ]

    new_X_data1 = np.concatenate([first_slice_X_data1, 
                                  second_slice_X_data2,
                                  third_slice_X_data3], 
                                  axis=0)
    new_X_data2 = np.concatenate([first_slice_X_data2, 
                                  second_slice_X_data3,
                                  third_slice_X_data1], 
                                  axis=0)
    new_X_data3 = np.concatenate([first_slice_X_data3, 
                                  second_slice_X_data1,
                                  third_slice_X_data2], 
                                  axis=0)

    new_y_data1 = np.concatenate([first_slice_y_data1, 
                                  second_slice_y_data2,
                                  third_slice_y_data3], 
                                  axis=0)
    new_y_data2 = np.concatenate([first_slice_y_data2, 
                                  second_slice_y_data3,
                                  third_slice_y_data1], 
                                  axis=0)
    new_y_data3 = np.concatenate([first_slice_y_data3, 
                                  second_slice_y_data1,
                                  third_slice_y_data2], 
                                  axis=0)

    # shuffle data
    # data1
    indexes = np.arange(new_X_data1.shape[0])
    np.random.shuffle(indexes)
    new_X_data1 = new_X_data1[indexes]
    new_y_data1 = new_y_data1[indexes]

    # data2
    indexes = np.arange(new_X_data2.shape[0])
    np.random.shuffle(indexes)
    new_X_data2 = new_X_data2[indexes]
    new_y_data2 = new_y_data2[indexes]

    # data3
    indexes = np.arange(new_X_data3.shape[0])
    np.random.shuffle(indexes)
    new_X_data3 = new_X_data3[indexes]
    new_y_data3 = new_y_data3[indexes]

    save_file1_path = file1_path.split('/')[-1]
    save_file2_path = file2_path.split('/')[-1]
    save_file3_path = file3_path.split('/')[-1]
    if save_dir is not None:
        save_file1_path = '{0}/{1}'.format(save_dir, save_file1_path)
        save_file2_path = '{0}/{1}'.format(save_dir, save_file2_path)
        save_file3_path = '{0}/{1}'.format(save_dir, save_file3_path)

    np.savez(save_file1_path, examples=new_X_data1, labels=new_y_data1)
    np.savez(save_file2_path, examples=new_X_data2, labels=new_y_data2)
    np.savez(save_file3_path, examples=new_X_data3, labels=new_y_data3)

    return


def main(args):
    if args.save_dir is not None:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

    if args.num_files == 2:
        # shuffling examples in two npz files
        two_split_mixer(args.npz_file1_path, args.npz_file2_path, args.save_dir)
    elif args.num_files == 3:
        # shuffling examples in three npz files
        three_split_mixer(args.npz_file1_path, args.npz_file2_path, args.npz_file3_path, args.save_dir)
    else:
        print('ERROR: Number of files to mix should be 2 or 3')
        print('Exiting')

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('num_files',
                        help='Number of npz files to mix. [i.e. 2 or 3]',
                        choices=[2,3],
                        type=int)

    parser.add_argument('npz_file1_path',
                        help='Path to the first npz file',
                        type=str)

    parser.add_argument('npz_file2_path',
                        help='Path to the second npz file',
                        type=str)

    parser.add_argument('--npz-file3-path',
                        help='Path to the third npz file',
                        type=str)
    parser.add_argument('--save-dir',
                        help='Directory to save generated files',
                        type=str)

    main(parser.parse_args())
