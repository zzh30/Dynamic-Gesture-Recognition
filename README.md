# 动态手势识别
## 主要方法
使用mediapipe提取21个手部骨架点信息作为输入，使用STGCN++进行动态手势识别模型的训练。

## 主要流程
![image](https://github.com/user-attachments/assets/c18fe2d2-15fb-4e83-9972-87db60c6b68a)

## 数据集
jester数据集：包含27种不同类别的手势，训练集、验证集、测试集分别由118562、14787、14743个视频组成。每个视频由37帧组成，fps为12。
### 下载链接
https://www.qualcomm.com/developer/software/jester-dataset
### 数据集示例
![image](https://github.com/user-attachments/assets/2af748db-1364-47be-b14c-a5f6f21c8b6c)
### 数据集细节
![image](https://github.com/user-attachments/assets/ee3fff9c-5b7b-4651-8b7d-83731c771574)

## 使用mediapipe提取21个手部骨架点

## 使用STGCN++进行动态手势识别模型的训练

## 实现手势识别

