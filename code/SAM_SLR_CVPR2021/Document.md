项目文件结构如下：
```
├─Conv3D
│  │  dataset_sign_clip.py                              数据集读取、划分
│  │  dataset_sign_flow_clip.py                         光流数据集读取、划分
│  │  readme.md
│  │  requirements.txt
│  │  Sign_Isolated_Conv3D_clip.py                      RGB分支
│  │  Sign_Isolated_Conv3D_clip_finetune.py
│  │  Sign_Isolated_Conv3D_clip_test.py
│  │  Sign_Isolated_Conv3D_depth_flow_clip.py           深度分支
│  │  Sign_Isolated_Conv3D_depth_flow_clip_finetune.py  
│  │  Sign_Isolated_Conv3D_depth_flow_clip_test.py
│  │  Sign_Isolated_Conv3D_flow_clip.py                 光流分支
│  │  Sign_Isolated_Conv3D_flow_clip_finetune.py
│  │  Sign_Isolated_Conv3D_flow_clip_test.py
│  │  Sign_Isolated_Conv3D_hha_clip_mask.py             HHA分支
│  │  Sign_Isolated_Conv3D_hha_clip_mask_finetune.py
│  │  Sign_Isolated_Conv3D_hha_clip_mask_test.py
│  │  tools.py
│  │  train.py                                          训练
│  │  utils.py
│  │  validation_clip.py                                验证
│  │
│  ├─data
│  │      test_labels_pseudo.csv
│  │      train_labels.csv
│  │      train_val_labels.csv
│  │      val_gt.csv
│  │
│  └─models
│          Conv3D.py
│
├─data-prepare  预处理各类输入
│  │  .gitattributes
│  │  gen_flow.py                               生成RGB、Depth数据流，需要修改folder, npy_folder, out_folder文件路径
│  │  gen_frames.py                             从视频获取RGB帧，需要修改folder, npy_folder, out_folder
│  │  gen_hha.py                                生成HHA，很慢。改变文件夹、npy_folder、out_folder变量
│  │  optical_flow_guidelines.docx
│  │  README.MD
│  │
│  ├─Depth2HHA
│  │  │  .gitattributes
│  │  │  camera_rotations_NYU.txt
│  │  │  CVPR21Chal_convert_HHA.m                   改变input_folder和output_folder以及hha_root变量
│  │  │  README.md
│  │  │  saveHHA.m
│  │  │
│  │  ├─demo-data
│  │  │      pd.txt
│  │  │      showpd.py
│  │  │
│  │  ├─unused
│  │  │      main.m
│  │  │      test.m
│  │  │
│  │
│  └─wholepose                              产生全身关键点，需要修改input_path and output_npy 
│      │  demo.py                           产生全身关键点
│      │  download_pretrained.txt
│      │  pose_hrnet.py
│      │  readme.md
│      │  requirements.txt
│      │  utils.py
│      │  wholebody_w48_384x288.yaml
│      │
│      └─config
│              default.py
│              models.py
│              __init__.py
│
├─ensemble  用于模型合并
│  │  ensemble_multimodal_rgb.py
│  │  ensemble_multimodal_rgbd.py
│  │  predictions_rgb.csv
│  │  predictions_rgbd.csv
│  │  readme.txt
│  │  test_feature_w_val_finetune.pkl
│  │  test_flow_color_w_val_finetune.pkl
│  │  test_flow_depth_w_val_finetune.pkl
│  │  test_gcn_w_val_finetune.pkl
│  │  test_hha_w_val_finetune.pkl
│  │  test_labels_pseudo.pkl
│  │  test_rgb_w_val_finetune.pkl
│  │
│  └─gcn
│          bone.pkl
│          bone_finetuned.pkl
│          bone_motion.pkl
│          bone_motion_finetuned.pkl
│          ensemble_finetune.py
│          ensemble_wo_val.py
│          joint.pkl
│          joint_finetuned.pkl
│          joint_motion.pkl
│          joint_motion_finetuned.pkl
│          readme.txt
│          test_labels_pseudo.pkl
│
├─img
│      AUTSL_test.jpg
│      AUTSL_val.jpg
│      challenge_result.jpg
│      RGBD_track.jpg
│      RGB_track.jpg
│      sam-slr.jpg
│      sam-slr_1280px.jpg
│      SLR500.jpg
│      WLASL2000.jpg
│
├─SL-GCN
│  │  LICENSE
│  │  main.py
│  │  readme.md
│  │
│  ├─config
│  │  └─sign
│  │      ├─finetune
│  │      │      train_bone.yaml
│  │      │      train_bone_motion.yaml
│  │      │      train_joint.yaml
│  │      │      train_joint_motion.yaml
│  │      │
│  │      ├─test
│  │      │      test_bone.yaml
│  │      │      test_bone_motion.yaml
│  │      │      test_joint.yaml
│  │      │      test_joint_motion.yaml
│  │      │
│  │      ├─test_finetuned
│  │      │      test_bone.yaml
│  │      │      test_bone_motion.yaml
│  │      │      test_joint.yaml
│  │      │      test_joint_motion.yaml
│  │      │
│  │      └─train
│  │              train_bone.yaml
│  │              train_bone_motion.yaml
│  │              train_joint.yaml
│  │              train_joint_motion.yaml
│  │
│  ├─data
│  │  └─sign
│  │      └─27_2
│  │              gen_train_val.py
│  │
│  ├─data_gen
│  │      gen_bone_data.py
│  │      gen_motion_data.py
│  │      sign_gendata.py
│  │      __init__.py
│  │
│  ├─feeders
│  │      feeder.py
│  │      tools.py
│  │      __init__.py
│  │
│  ├─graph
│  │      sign_27.py
│  │      tools.py
│  │      __init__.py
│  │
│  └─model
│          decouple_gcn_attn.py
│          dropSke.py
│          dropT.py
│          __init__.py
│
└─SSTCN
    │  load_newfeature.py
    │  main_process.sh
    │  readme.txt
    │  test.py
    │  test_labels_pseudo.pkl
    │  train_parallel.py                        执行训练
    │  train_val_split.mat
    │  T_Pose_model.py
    │
    └─data_process
        │  download_pretrained_model.txt
        │  pose_hrnet.py
        │  wholebody_w48_384x384_adam_lr1e-3.yaml
        │  wholepose_features_extraction.py                提取骨架信息
        │
        └─config
                default.py
                models.py
                __init__.py

```