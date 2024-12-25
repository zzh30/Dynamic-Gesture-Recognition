import cv2
import numpy as np
import mediapipe as mp
from operator import itemgetter
from mmaction.apis import init_recognizer, inference_recognizer

# 用于提取关键点的MediaPipe库
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# 输入视频路径
video_path = '/media/disk2/zzh/mmaction2-main/2537.mp4'
cap = cv2.VideoCapture(video_path)

# 视频信息
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
img_shape = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))

# 创建字典存储提取到的信息
annotations = {
    'frame_dir': 'Train/demo1',  # 可以设置为视频的标识符或路径
    'label': -1,  # 动作标签，默认为 -1 表示未定义
    'img_shape': img_shape,
    'original_shape': img_shape,
    'total_frames': total_frames,
    'keypoint': [],  # 存储每帧的关键点
    'keypoint_score': []  # 存储关键点的置信度
}

frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 将BGR图像转换为RGB图像以便MediaPipe处理
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 进行手部关键点检测
    results = hands.process(rgb_frame)

    # 如果检测到手
    if results.multi_hand_landmarks:
        keypoints = []
        scores = []
        for hand_landmarks in results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                # 将关键点的x、y坐标添加到keypoints列表（去掉z轴）
                keypoints.append([landmark.x, landmark.y])  # 只保留归一化的 x, y 坐标
                scores.append(1.0)  # 使用固定置信度 1.0

        # 将每帧的关键点和置信度添加到annotations字典中
        annotations['keypoint'].append(np.array(keypoints))
        annotations['keypoint_score'].append(np.array(scores))
    else:
        # 如果没有检测到手，添加全零数组表示没有关键点
        annotations['keypoint'].append(np.zeros((21, 2)))  # 假设有21个关节点 (x, y)
        annotations['keypoint_score'].append(np.zeros(21))

    frame_idx += 1

cap.release()

# 检查提取到的帧数是否与总帧数一致，不一致则补齐
if len(annotations['keypoint']) < total_frames:
    missing_frames = total_frames - len(annotations['keypoint'])
    for _ in range(missing_frames):
        annotations['keypoint'].append(np.zeros((21, 2)))  # 补充全零关键点
        annotations['keypoint_score'].append(np.zeros(21))

# 将关键点数据转换为ndarray，以符合要求的格式
annotations['keypoint'] = np.array([annotations['keypoint']])  # 形状为 (M, T, V, C)，这里 M=1
annotations['keypoint_score'] = np.array([annotations['keypoint_score']])  # 形状为 (M, T, V)

# 配置文件路径
config_file = '/media/disk2/zzh/mmaction2-main/projects/gesture_recognition/configs/stgcnpp_8xb16-joint-u100-16e_jester-keypoint-2d.py'
checkpoint_file = '/media/disk2/zzh/mmaction2-main/work_dirs/stgcnpp_8xb16-joint-u100-16e_jester-keypoint-2d/best_acc_top1_epoch_15.pth'

# 加载模型
model = init_recognizer(config_file, checkpoint_file, device='cpu')

# 使用提取的关键点信息进行推理
results = inference_recognizer(model, annotations)

# 获取预测分数
pred_scores = results.pred_score.tolist()
score_tuples = tuple(zip(range(len(pred_scores)), pred_scores))
score_sorted = sorted(score_tuples, key=itemgetter(1), reverse=True)
top5_label = score_sorted[:5]

# 读取标签并显示结果
label = '/media/disk2/zzh/mmaction2-main/label.txt'
labels = open(label).readlines()
labels = [x.strip() for x in labels]
results = [(labels[k[0]], k[1]) for k in top5_label]

for result in results:
    print(f'{result[0]}: ', result[1])
