# Training details
The training process is improved based on the model files from MMAction2. First, download the [MMAction2](https://github.com/open-mmlab/mmaction2/tree/main) model files to the `$MMAction` directory. Then, replace the following files in `$MMAction` with the corresponding files from this folder:

`$MMAction`/projects/gesture_recognition/configs/stgcnpp_8xb16-joint-u100-16e_jester-keypoint-2d.py  
`$MMAction`/mmaction/models/utils/graph.py  
`$MMAction`/mmaction/datasets/transforms/pose_transforms.py

Train using the following method.
```bash
bash tools/dist_train.sh $MMAction/projects/gesture_recognition/configs/stgcnpp_8xb16-joint-u100-80e_jester-keypoint-2d.py 8
```
