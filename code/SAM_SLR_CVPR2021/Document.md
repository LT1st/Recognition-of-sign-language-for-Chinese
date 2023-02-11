��Ŀ�ļ��ṹ���£�
```
����Conv3D
��  ��  dataset_sign_clip.py                              ���ݼ���ȡ������
��  ��  dataset_sign_flow_clip.py                         �������ݼ���ȡ������
��  ��  readme.md
��  ��  requirements.txt
��  ��  Sign_Isolated_Conv3D_clip.py                      RGB��֧
��  ��  Sign_Isolated_Conv3D_clip_finetune.py
��  ��  Sign_Isolated_Conv3D_clip_test.py
��  ��  Sign_Isolated_Conv3D_depth_flow_clip.py           ��ȷ�֧
��  ��  Sign_Isolated_Conv3D_depth_flow_clip_finetune.py  
��  ��  Sign_Isolated_Conv3D_depth_flow_clip_test.py
��  ��  Sign_Isolated_Conv3D_flow_clip.py                 ������֧
��  ��  Sign_Isolated_Conv3D_flow_clip_finetune.py
��  ��  Sign_Isolated_Conv3D_flow_clip_test.py
��  ��  Sign_Isolated_Conv3D_hha_clip_mask.py             HHA��֧
��  ��  Sign_Isolated_Conv3D_hha_clip_mask_finetune.py
��  ��  Sign_Isolated_Conv3D_hha_clip_mask_test.py
��  ��  tools.py
��  ��  train.py                                          ѵ��
��  ��  utils.py
��  ��  validation_clip.py                                ��֤
��  ��
��  ����data
��  ��      test_labels_pseudo.csv
��  ��      train_labels.csv
��  ��      train_val_labels.csv
��  ��      val_gt.csv
��  ��
��  ����models
��          Conv3D.py
��
����data-prepare  Ԥ�����������
��  ��  .gitattributes
��  ��  gen_flow.py                               ����RGB��Depth����������Ҫ�޸�folder, npy_folder, out_folder�ļ�·��
��  ��  gen_frames.py                             ����Ƶ��ȡRGB֡����Ҫ�޸�folder, npy_folder, out_folder
��  ��  gen_hha.py                                ����HHA���������ı��ļ��С�npy_folder��out_folder����
��  ��  optical_flow_guidelines.docx
��  ��  README.MD
��  ��
��  ����Depth2HHA
��  ��  ��  .gitattributes
��  ��  ��  camera_rotations_NYU.txt
��  ��  ��  CVPR21Chal_convert_HHA.m                   �ı�input_folder��output_folder�Լ�hha_root����
��  ��  ��  README.md
��  ��  ��  saveHHA.m
��  ��  ��
��  ��  ����demo-data
��  ��  ��      pd.txt
��  ��  ��      showpd.py
��  ��  ��
��  ��  ����unused
��  ��  ��      main.m
��  ��  ��      test.m
��  ��  ��
��  ��
��  ����wholepose                              ����ȫ��ؼ��㣬��Ҫ�޸�input_path and output_npy 
��      ��  demo.py                           ����ȫ��ؼ���
��      ��  download_pretrained.txt
��      ��  pose_hrnet.py
��      ��  readme.md
��      ��  requirements.txt
��      ��  utils.py
��      ��  wholebody_w48_384x288.yaml
��      ��
��      ����config
��              default.py
��              models.py
��              __init__.py
��
����ensemble  ����ģ�ͺϲ�
��  ��  ensemble_multimodal_rgb.py
��  ��  ensemble_multimodal_rgbd.py
��  ��  predictions_rgb.csv
��  ��  predictions_rgbd.csv
��  ��  readme.txt
��  ��  test_feature_w_val_finetune.pkl
��  ��  test_flow_color_w_val_finetune.pkl
��  ��  test_flow_depth_w_val_finetune.pkl
��  ��  test_gcn_w_val_finetune.pkl
��  ��  test_hha_w_val_finetune.pkl
��  ��  test_labels_pseudo.pkl
��  ��  test_rgb_w_val_finetune.pkl
��  ��
��  ����gcn
��          bone.pkl
��          bone_finetuned.pkl
��          bone_motion.pkl
��          bone_motion_finetuned.pkl
��          ensemble_finetune.py
��          ensemble_wo_val.py
��          joint.pkl
��          joint_finetuned.pkl
��          joint_motion.pkl
��          joint_motion_finetuned.pkl
��          readme.txt
��          test_labels_pseudo.pkl
��
����img
��      AUTSL_test.jpg
��      AUTSL_val.jpg
��      challenge_result.jpg
��      RGBD_track.jpg
��      RGB_track.jpg
��      sam-slr.jpg
��      sam-slr_1280px.jpg
��      SLR500.jpg
��      WLASL2000.jpg
��
����SL-GCN
��  ��  LICENSE
��  ��  main.py
��  ��  readme.md
��  ��
��  ����config
��  ��  ����sign
��  ��      ����finetune
��  ��      ��      train_bone.yaml
��  ��      ��      train_bone_motion.yaml
��  ��      ��      train_joint.yaml
��  ��      ��      train_joint_motion.yaml
��  ��      ��
��  ��      ����test
��  ��      ��      test_bone.yaml
��  ��      ��      test_bone_motion.yaml
��  ��      ��      test_joint.yaml
��  ��      ��      test_joint_motion.yaml
��  ��      ��
��  ��      ����test_finetuned
��  ��      ��      test_bone.yaml
��  ��      ��      test_bone_motion.yaml
��  ��      ��      test_joint.yaml
��  ��      ��      test_joint_motion.yaml
��  ��      ��
��  ��      ����train
��  ��              train_bone.yaml
��  ��              train_bone_motion.yaml
��  ��              train_joint.yaml
��  ��              train_joint_motion.yaml
��  ��
��  ����data
��  ��  ����sign
��  ��      ����27_2
��  ��              gen_train_val.py
��  ��
��  ����data_gen
��  ��      gen_bone_data.py
��  ��      gen_motion_data.py
��  ��      sign_gendata.py
��  ��      __init__.py
��  ��
��  ����feeders
��  ��      feeder.py
��  ��      tools.py
��  ��      __init__.py
��  ��
��  ����graph
��  ��      sign_27.py
��  ��      tools.py
��  ��      __init__.py
��  ��
��  ����model
��          decouple_gcn_attn.py
��          dropSke.py
��          dropT.py
��          __init__.py
��
����SSTCN
    ��  load_newfeature.py
    ��  main_process.sh
    ��  readme.txt
    ��  test.py
    ��  test_labels_pseudo.pkl
    ��  train_parallel.py                        ִ��ѵ��
    ��  train_val_split.mat
    ��  T_Pose_model.py
    ��
    ����data_process
        ��  download_pretrained_model.txt
        ��  pose_hrnet.py
        ��  wholebody_w48_384x384_adam_lr1e-3.yaml
        ��  wholepose_features_extraction.py                ��ȡ�Ǽ���Ϣ
        ��
        ����config
                default.py
                models.py
                __init__.py

```