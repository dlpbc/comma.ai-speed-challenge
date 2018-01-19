# comma.ai speed challenge
The aim of the challenge is to predict the speed of a car from pre-recorded video frames (training and test video data) provided.  

[Challenge Link](https://twitter.com/comma_ai/status/849131721572327424)

## Challenge Description
Description by [comma.ai](https://comma.ai)  

```
Welcome to the comma.ai 2017 Programming Challenge!

Basically, your goal is to predict the speed of a car from a video.

data/train.mp4 is a video of driving containing 20400 frames. Video is shot at 20 fps.
data/train.txt contains the speed of the car at each frame, one speed on each line.

data/test.mp4 is a different driving video containing 10798 frames. Video is shot at 20 fps.
Your deliverable is test.txt

We will evaluate your test.txt using mean squared error. <10 is good. <5 is better. <3 is heart.
```

## Model Employed
Applied transfer learning on [Inflated 3d (I3d) inception](https://github.com/dlpbc/keras-kinetics-i3d) architecture (without the original classification layer) pretrained on imagenet and kinetics datasets. The pretrained weights was used to initialise the model and the model was trained end-to-end.

## Requirements
- keras
- python-opencv (cv2)
- python3

## Data Preparation
The train video data is split into training and valiation videos using 70:30 split rule. Additionally, the videos are chopped into example clips (each clip contains 40 frames) and stored in a compressed numpy file (npz). Each clip can contain frames that overlaps with frames in another clip.  
To prepate data, use the command below

```
sh prepare_data.sh
```
The data preparation on my local machine took about 53 minutes, with large bulk of the time (about 49 minutes) taken for conversion of the (RGB) video to Optical Flow video.

## Training
To train the RGB model, use the comand below
```
python3 rgb_train.py
```

To train the Optical Flow model, use the command below
```
python3 flow_train.py
```

