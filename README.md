# Dynamic Gesture Recognition
## Main methods
Use MediaPipe to extract 21 hand skeleton points as input, and train a dynamic gesture recognition model using STGCN++.

## Main process
![image](https://github.com/user-attachments/assets/c18fe2d2-15fb-4e83-9972-87db60c6b68a)

## Dataset
[Jester dataset](https://www.qualcomm.com/developer/software/jester-dataset): It contains 27 different categories of gestures, with the training set, validation set, and test set consisting of 118,562, 14,787, and 14,743 videos respectively. Each video consists of 37 frames and has a frame rate of 12 fps.
### Sample
![image](https://github.com/user-attachments/assets/2af748db-1364-47be-b14c-a5f6f21c8b6c)
### Details
![image](https://github.com/user-attachments/assets/ee3fff9c-5b7b-4651-8b7d-83731c771574)  

You need to download the annotation file yourself and process the dataset into the following format.

```
jester
|── data
    |-- Train
       │-- Zooming Out With Two Fingers
	  │-- 00010.mp4
	  │-- 00012.mp4
	  │-- ..
       │-- Turning Hand Clockwise
	  │-- 00123.mp4
	  │-- 00341.mp4
	  │-- ..
       │-- ..
    |-- Validation
       │-- Zooming Out With Two Fingers
	  │-- 00100.mp4
	  │-- 00102.mp4
	  │-- ..
       │-- Turning Hand Clockwise
	  │-- 00123.mp4
	  │-- 00341.mp4
	  │-- ..
       │-- ..
```
## Use MediaPipe to extract 21 hand skeleton points.
Use [extract-key.py](https://github.com/zzh30/Dynamic-Gesture-Recognition/blob/main/demo/inference.py) to extract keypoint information from videos and save it as a `.pkl` file.

```
python extract_keypoints.py ./dataset ./output_keypoints.pkl
```
## Train a model 
Use STGCN++ to [train](https://github.com/zzh30/Dynamic-Gesture-Recognition/tree/main/train) the gesture recognition model.
## Demo
Use the [inference.py](https://github.com/zzh30/Dynamic-Gesture-Recognition/blob/main/demo/inference.py) to access the camera and perform real-time gesture recognition. The results are as follows:
![demo](https://github.com/user-attachments/assets/5c36afba-6767-452b-b34f-e2a346b7faf1)

