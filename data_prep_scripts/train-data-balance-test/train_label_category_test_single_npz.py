'''
this script loads the labels from an npz files rather than a single txt file
'''
import os
import argparse
import numpy as np

def main(args):

    # load the label files
    npz_file = np.load(args.npz_path)
    labels = npz_file['labels'].reshape((npz_file['labels'].shape[0], )).tolist()

    if args.category == 'coarse': # group the labels in range of 5's
        range_count1 = 0 # 0 - 5
        range_count2 = 0 # 6 - 10
        range_count3 = 0 # 11 - 15
        range_count4 = 0 # 16 - 20
        range_count5 = 0 # 21 - 25
        range_count6 = 0 # others (>25) 

        for label in labels:
            label = int(label)
            if label <= 5:
                range_count1 +=1

            elif label >= 6 and label <= 10:
                range_count2 += 1

            elif label >= 11 and label <= 15:
                range_count3 += 1

            elif label >= 16 and label <= 20:
                range_count4 += 1

            elif label >= 21 and label <= 25:
                range_count5 += 1

            else:
                range_count6 += 1

        print('Results for range count...\n')
        print('Num of labels in range 0 - 5   is %d' % range_count1)
        print('Num of labels in range 6 - 10  is %d' % range_count2)
        print('Num of labels in range 11 - 15 is %d' % range_count3)
        print('Num of labels in range 16 - 20 is %d' % range_count4)
        print('Num of labels in range 21 - 25 is %d' % range_count5)
        print('Num of labels in range > 25    is %d' % range_count6)

    else: # group the labels in ange of 2's
        categories = np.zeros((15, ))

        for label in labels:
            label = int(label)
            if label >= 0  and label <= 2:  categories[0]  += 1
            if label >= 3  and label <= 4:  categories[1]  += 1
            if label >= 5  and label <= 6:  categories[2]  += 1
            if label >= 7  and label <= 8:  categories[3]  += 1
            if label >= 9  and label <= 10: categories[4]  += 1
            if label >= 11 and label <= 12: categories[5]  += 1
            if label >= 13 and label <= 14: categories[6]  += 1
            if label >= 15 and label <= 16: categories[7]  += 1
            if label >= 17 and label <= 18: categories[8]  += 1
            if label >= 19 and label <= 20: categories[9]  += 1
            if label >= 21 and label <= 22: categories[10] += 1
            if label >= 23 and label <= 24: categories[11] += 1
            if label >= 25 and label <= 26: categories[12] += 1
            if label >= 27 and label <= 28: categories[13] += 1
            if label >= 29 and label <= 30: categories[14] += 1


        print('Results for range count...\n')
        print('Num of labels in range 0 - 2  is %d' % categories[0])
        i = 3
        for category in categories[1:]:
            print('Num of labels in range %d - %d  is %d' % (i, i+1, category))
            i += 2

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('npz_path', help='path to npz file')
    parser.add_argument('-c', '--category', 
                        help='select category type. coarse = grouping of 5, fine = groupings of 2.' 
                              'Note: this excludes the first group where its 6 for coarse and 2 for'
                              'fine', 
                        type=str,
                        choices=['coarse', 'fine'],
                        default='coarse')

    main(parser.parse_args())
