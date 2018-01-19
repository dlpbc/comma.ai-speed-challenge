'''
Converts an RGB video to an optical flow based video
'''
import sys
import time
import argparse
import cv2
import numpy as np

FPS = 20.0 # 20 frames per second
def main(args):
    if args.rgb_label_path is not None and args.flow_label_path is None:
        print('ERROR:')
        print('You cannot set `--rgb-label-path` and leave out `--flow-label-path`')
        print('ERROR: both `--rgb-label-path` and `--flow-label-path` must be set')
        print('Exiting...')
        return
    if args.rgb_label_path is None and args.flow_label_path is not None:
        print('ERROR:')
        print('You cannot set `--flow-label-path` and leave out `--rgb-label-path`')
        print('ERROR: both `--rgb-label-path` and `--flow-label-path` must be set')
        print('Exiting...')
        return
    
    print('INFO: this will take some time')
    print('converting rgb video to optical flow video')
    print('RGB video path: %s' % args.rgb_video_path)
    print('(Optical Flow) output path: %s' % args.flow_video_output_path)
    print('using %s algorithm\n' % args.optical_flow_algorithm)
    # open RGB video for reading
    cap = cv2.VideoCapture(args.rgb_video_path)
    num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    if args.rgb_label_path is not None:
        # retrieve labels
        f = open(args.rgb_label_path, 'r')
        labels = f.readlines()
        f.close()
    
        # sanity check
        if num_frames != len(labels):
            print('ERROR:') 
            print('number of frames in video does NOT match number of label(rows) in label file...')
            print('please ensure they match...')
            print('Exiting...')
            return

        # open flow label file for writing
        f = open(args.flow_label_path, 'w')
        labels = labels[1:] # optical flow frames/labels starts from RGB second frame to the end
        f.writelines(labels)
        f.close()


    # open output (optical flow) video for writing
    video_compression_codec = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(args.flow_video_output_path,
        video_compression_codec,
        FPS,
        frame_size)

    # select optical flow algorithm to use
    if args.optical_flow_algorithm == 'tvl1':
        tvl1 = cv2.DualTVL1OpticalFlow_create() # we will need this in the loop 
 
    t1 = time.time()
    ret, prev_frame = cap.read()
    prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(prev_frame)

    frame_cnt = 1
    while True:
        ret, next_frame = cap.read()
        if next_frame is None:
            break

        next_frame_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

        if args.optical_flow_algorithm == 'farneback':
            flow = cv2.calcOpticalFlowFarneback(prev_frame_gray, next_frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        else: # tvl1
            flow = tvl1.calc(prev_frame_gray, next_frame_gray, None)

        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        hsv[...,0] = ang*180/np.pi/2
        hsv[...,1] = 255
        hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        bgr_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        video_writer.write(bgr_flow)
        prev_frame_gray = next_frame_gray

        frame_cnt += 1
        sys.stdout.write('\rprocessed frames: %d of %d' % (frame_cnt, num_frames))

    t2 = time.time()
    cap.release()
    video_writer.release()
    print('\nconversion completed...')
    print('time taken: %d seconds\n\n' % (t2 - t1))
    return
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('rgb_video_path', help='Input RGB video file path')
    parser.add_argument('flow_video_output_path', help='Output (Optical Flow) video file path')

    parser.add_argument('--rgb-label-path', 
                        help='Input label path for frames in RGB video. If specified,' 
                             '`--flow-label-path` must also be specified .')

    parser.add_argument('--flow-label-path', 
                        help='Output label path for frames in optical flow video. If specified,'
                        '`--rgb-label-path` must also be specified.')

    parser.add_argument('-t', '--optical-flow-algorithm', 
                        help='specify the (dense) optical flow algorithm to use',
                        type=str,
                        choices=['farneback', 'tvl1'],
                        default='farneback')
    main(parser.parse_args())
