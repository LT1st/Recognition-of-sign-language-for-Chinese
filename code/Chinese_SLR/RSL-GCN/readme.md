# Skeleton Based Sign Language Recognition
先运行data_gen；数据是/data/sign/里面的（来自于数据预处理）
再运行main
## Data preparation
1. Extract whole-body keypoints data following the instruction in ../data_process/wholepose
2. Run the following code to prepare the data for GCN.

        cd data_gen/
        python sign_gendata.py
        python gen_bone_data.py
        python gen_motion.py
## Usage
### Train:
```
python main.py --config config/sign/train/train_joint.yaml

python main.py --config config/sign/train/train_bone.yaml

python main.py --config config/sign/train/train_joint_motion.yaml

python main.py --config config/sign/train/train_bone_motion.yaml
```
### Finetune:
```
python main.py --config config/sign/finetune/train_joint.yaml

python main.py --config config/sign/finetune/train_bone.yaml

python main.py --config config/sign/finetune/train_joint_motion.yaml

python main.py --config config/sign/finetune/train_bone_motion.yaml
```
### Test:
```
python main.py --config config/sign/test/test_joint.yaml

python main.py --config config/sign/test/test_bone.yaml

python main.py --config config/sign/test/test_joint_motion.yaml

python main.py --config config/sign/test/test_bone_motion.yaml
```

1. Copy the results .pkl files from all streams (joint, bone, joint motion and bone motion) to ../ensemble/gcn and renamed them correctly.
2. Follow the instruction in ../ensemble/gcn to obtained the results of multi-stream ensemble.