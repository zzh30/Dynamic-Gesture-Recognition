import os
import pickle
import time
from argparse import ArgumentParser
import mediapipe as mp
import cv2
import numpy as np

# Parse command line arguments
def parse_args():
    parser = ArgumentParser()
    parser.add_argument('root', help='Root directory containing video files')  # Root directory containing video files
    parser.add_argument('output', help='Output pickle file path')  # Path to save output pickle file
    args = parser.parse_args()
    return args

# Define label mapping based on the image you provided
label_map = {
    'Stop Sign': 0,
    'Turning Hand Counterclockwise': 1,
    'Shaking Hand': 2,
    'Sliding Two Fingers Left': 3,
    'Swiping Down': 4,
    'Pushing Hand Away': 5,
    'Zooming Out With Two Fingers': 6,
    'Swiping Right': 7,
    'Pulling Two Fingers In': 8,
    'Pushing Two Fingers Away': 9,
    'Doing other things': 10,
    'Drumming Fingers': 11,
    'Rolling Hand Backward': 12,
    'Swiping Left': 13,
    'Rolling Hand Forward': 14,
    'Sliding Two Fingers Right': 15,
    'Thumb Up': 16,
    'Zooming In With Full Hand': 17,
    'Thumb Down': 18,
    'Turning Hand Clockwise': 19,
    'No gesture': 20,
    'Sliding Two Fingers Down': 21,
    'Zooming In With Two Fingers': 22,
    'Swiping Up': 23,
    'Zooming Out With Full Hand': 24,
    'Pulling Hand In': 25,
    'Sliding Two Fingers Up': 26
}

# Extract hand keypoints from video folder using MediaPipe
def extract_keypoints_mediapipe(video_path):
    mp_hands = mp.solutions.hands 
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
    cap = cv2.VideoCapture(video_path) 
    
    keypoints = [] 
    keypoint_scores = [] 
    img_shape = None  

    frame_count = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        if frame is None:
            continue  

        # If image shape has not been set, get the shape from the first frame
        if img_shape is None:
            img_shape = frame.shape[0:2]

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
        results = hands.process(image_rgb) 

        frame_count += 1
        if frame_count % 50 == 0:
            print(f'Processing frame {frame_count} in video {video_path}...')

        if results.multi_hand_landmarks:
            kp = []
            scores = []
            for hand_landmarks in results.multi_hand_landmarks:
                # Extract keypoints for each detected hand
                for landmark in hand_landmarks.landmark:
                    kp.append([landmark.x, landmark.y])  
                    scores.append(1.0)  
            keypoints.append(np.array(kp))  
            keypoint_scores.append(np.array(scores))  
        else:
            # Store zero keypoints and confidence scores if no hand is detected
            keypoints.append(np.zeros((21, 2)))  
            keypoint_scores.append(np.zeros(21))  

    cap.release()  
    hands.close() 
    
    if img_shape is None:
        img_shape = (0, 0)

    return keypoints, keypoint_scores, len(keypoints), img_shape

def main():
    args = parse_args()

    annotations = []  
    L = 0  
    t = time.time() 

    # Count total number of videos for ETA calculation
    total_videos = sum(len(files) for _, _, files in os.walk(args.root) if files)
    
    # Iterate through all train and validation folders
    for split in ['Train', 'Validation']:
        split_dir = os.path.join(args.root, split)
        if not os.path.exists(split_dir):
            continue
        for label in os.listdir(split_dir):
            label_dir = os.path.join(split_dir, label)
            if os.path.isdir(label_dir):
                for video_file in os.listdir(label_dir):
                    video_path = os.path.join(label_dir, video_file)
                    if os.path.isfile(video_path) and video_path.endswith(('.mp4', '.avi', '.mov')):
                        print(f'Starting processing of video: {video_path}')
                        keypoints, keypoint_scores, total_frames, img_shape = extract_keypoints_mediapipe(video_path)  
                        print(f'Finished processing of video: {video_path} with {total_frames} frames.')

                        # Get the integer label from label_map
                        int_label = label_map.get(label, -1)  

                        # Format the annotation as per the required structure
                        annotation = {
                            'frame_dir': f'{split}/{label}/{video_file}',
                            'label': int_label,  
                            'img_shape': img_shape,
                            'original_shape': img_shape,
                            'total_frames': total_frames,
                            'keypoint': np.array([keypoints]),  # 4D array: [M x T x V x C], M=1
                            'keypoint_score': np.array([keypoint_scores])  # 3D array: [M x T x V], M=1
                        }
                        annotations.append(annotation)
                        L += 1
                        
                        # Print ETA every 100 video files processed
                        if L % 100 == 0:
                            elapsed_time = time.time() - t
                            avg_time_per_video = elapsed_time / L
                            remaining_videos = total_videos - L
                            eta = (avg_time_per_video * remaining_videos) / 3600  
                            print('Estimated time remaining: %.2f hours' % eta)

    # Define split field for training and validation data
    split = {
        'train': [annotation['frame_dir'] for annotation in annotations if 'Train' in annotation['frame_dir']],
        'val': [annotation['frame_dir'] for annotation in annotations if 'Validation' in annotation['frame_dir']]
    }

    data = {
        'split': split,
        'annotations': annotations
    }

    # Save the extracted keypoints and metadata to a pickle file
    with open(args.output, 'wb') as f:
        pickle.dump(data, f)
    print(f'All keypoints saved to {args.output}')

if __name__ == '__main__':
    main()
