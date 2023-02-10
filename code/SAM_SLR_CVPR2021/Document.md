├─Conv3D
│  │  dataset_sign_clip.py                              数据集切片
│  │  dataset_sign_flow_clip.py                         数据集
│  │  readme.md
│  │  requirements.txt
│  │  Sign_Isolated_Conv3D_clip.py
│  │  Sign_Isolated_Conv3D_clip_finetune.py
│  │  Sign_Isolated_Conv3D_clip_test.py
│  │  Sign_Isolated_Conv3D_depth_flow_clip.py
│  │  Sign_Isolated_Conv3D_depth_flow_clip_finetune.py
│  │  Sign_Isolated_Conv3D_depth_flow_clip_test.py
│  │  Sign_Isolated_Conv3D_flow_clip.py
│  │  Sign_Isolated_Conv3D_flow_clip_finetune.py
│  │  Sign_Isolated_Conv3D_flow_clip_test.py
│  │  Sign_Isolated_Conv3D_hha_clip_mask.py
│  │  Sign_Isolated_Conv3D_hha_clip_mask_finetune.py
│  │  Sign_Isolated_Conv3D_hha_clip_mask_test.py
│  │  tools.py
│  │  train.py
│  │  utils.py
│  │  validation_clip.py
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
│  │  gen_flow.py
│  │  gen_frames.py
│  │  gen_hha.py
│  │  optical_flow_guidelines.docx
│  │  README.MD
│  │
│  ├─Depth2HHA
│  │  │  .gitattributes
│  │  │  camera_rotations_NYU.txt
│  │  │  CVPR21Chal_convert_HHA.m
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
│  │  └─utils
│  │      ├─depth_features
│  │      │  │  depthFeatures.m
│  │      │  │  processDepthImage.m
│  │      │  │
│  │      │  └─rgbdutils
│  │      │      │  computeNormals.m
│  │      │      │  computeNormalsSquareSupport.m
│  │      │      │  depthCuesHelper.m
│  │      │      │  fillHoles.m
│  │      │      │  getPointCloudFromZ.m
│  │      │      │  getRMatrix.m
│  │      │      │  getRMatrix2.m
│  │      │      │  getYDir.m
│  │      │      │  isRead.m
│  │      │      │  isWrite.m
│  │      │      │  jointBilateral.m
│  │      │      │  jointBilateralFile.m
│  │      │      │  LICENSE
│  │      │      │  README.md
│  │      │      │  rotatePC.m
│  │      │      │  startup_rgbdutils.m
│  │      │      │  wrapperComputeNormals.m
│  │      │      │  yuv2rgb_simple.m
│  │      │      │
│  │      │      ├─demo-rgbdutils
│  │      │      │      examples.m
│  │      │      │
│  │      │      └─imagestack
│  │      │          │  .gitignore
│  │      │          │  configure
│  │      │          │  design.txt
│  │      │          │  license.txt
│  │      │          │  Makefile
│  │      │          │  Makefile.common
│  │      │          │  TODO
│  │      │          │
│  │      │          ├─deps
│  │      │          │  └─OSX
│  │      │          │          SDLMain.h
│  │      │          │          SDLMain.m
│  │      │          │
│  │      │          ├─docs
│  │      │          │      header.inc
│  │      │          │      index.html
│  │      │          │      make-docs.sh
│  │      │          │      make-wiki.sh
│  │      │          │
│  │      │          ├─examples
│  │      │          │      life.sh
│  │      │          │      mandelbrot.sh
│  │      │          │
│  │      │          ├─makefiles
│  │      │          │      Makefile.cygwin
│  │      │          │      Makefile.linux
│  │      │          │      Makefile.macports
│  │      │          │
│  │      │          ├─matlab
│  │      │          │      joint_bilateral_mex.cpp
│  │      │          │      test_matlab.m
│  │      │          │
│  │      │          ├─msvc
│  │      │          │  │  ImageStack.sln
│  │      │          │  │  ImageStack.suo
│  │      │          │  │  ImageStack.vcproj
│  │      │          │  │  ImageStack.vcxproj
│  │      │          │  │  ImageStack.vcxproj.filters
│  │      │          │  │  ImageStack.vcxproj.user
│  │      │          │  │
│  │      │          │  ├─Example
│  │      │          │  │      Example.cpp
│  │      │          │  │      Example.vcxproj
│  │      │          │  │      Example.vcxproj.filters
│  │      │          │  │      Example.vcxproj.user
│  │      │          │  │
│  │      │          │  ├─ImageStack.lib
│  │      │          │  │      ImageStack.lib.vcxproj
│  │      │          │  │      ImageStack.lib.vcxproj.filters
│  │      │          │  │      ImageStack.lib.vcxproj.user
│  │      │          │  │
│  │      │          │  ├─include
│  │      │          │  │  │  api.h
│  │      │          │  │  │  cderror.h
│  │      │          │  │  │  cdjpeg.h
│  │      │          │  │  │  crc32.h
│  │      │          │  │  │  deflate.h
│  │      │          │  │  │  f77funcs.h
│  │      │          │  │  │  fftw3.h
│  │      │          │  │  │  guru.h
│  │      │          │  │  │  guru64.h
│  │      │          │  │  │  gzguts.h
│  │      │          │  │  │  inffast.h
│  │      │          │  │  │  inffixed.h
│  │      │          │  │  │  inflate.h
│  │      │          │  │  │  inftrees.h
│  │      │          │  │  │  jconfig.h
│  │      │          │  │  │  jdct.h
│  │      │          │  │  │  jerror.h
│  │      │          │  │  │  jinclude.h
│  │      │          │  │  │  jmemsys.h
│  │      │          │  │  │  jmorecfg.h
│  │      │          │  │  │  jpegint.h
│  │      │          │  │  │  jpeglib.h
│  │      │          │  │  │  jversion.h
│  │      │          │  │  │  mktensor-iodims.h
│  │      │          │  │  │  plan-guru-dft-c2r.h
│  │      │          │  │  │  plan-guru-dft-r2c.h
│  │      │          │  │  │  plan-guru-dft.h
│  │      │          │  │  │  plan-guru-r2r.h
│  │      │          │  │  │  plan-guru-split-dft-c2r.h
│  │      │          │  │  │  plan-guru-split-dft-r2c.h
│  │      │          │  │  │  plan-guru-split-dft.h
│  │      │          │  │  │  png.h
│  │      │          │  │  │  pngconf.h
│  │      │          │  │  │  t4.h
│  │      │          │  │  │  tiff.h
│  │      │          │  │  │  tiffconf.h
│  │      │          │  │  │  tiffio.h
│  │      │          │  │  │  tiffiop.h
│  │      │          │  │  │  tiffvers.h
│  │      │          │  │  │  tif_dir.h
│  │      │          │  │  │  tif_fax3.h
│  │      │          │  │  │  tif_predict.h
│  │      │          │  │  │  transupp.h
│  │      │          │  │  │  trees.h
│  │      │          │  │  │  uvcode.h
│  │      │          │  │  │  zconf.h
│  │      │          │  │  │  zlib.h
│  │      │          │  │  │  zutil.h
│  │      │          │  │  │
│  │      │          │  │  └─SDL
│  │      │          │  │          begin_code.h
│  │      │          │  │          close_code.h
│  │      │          │  │          doxyfile
│  │      │          │  │          SDL.h
│  │      │          │  │          SDL_active.h
│  │      │          │  │          SDL_audio.h
│  │      │          │  │          SDL_byteorder.h
│  │      │          │  │          SDL_cdrom.h
│  │      │          │  │          SDL_config.h
│  │      │          │  │          SDL_config.h.default
│  │      │          │  │          SDL_config.h.in
│  │      │          │  │          SDL_config_dreamcast.h
│  │      │          │  │          SDL_config_macos.h
│  │      │          │  │          SDL_config_macosx.h
│  │      │          │  │          SDL_config_minimal.h
│  │      │          │  │          SDL_config_nds.h
│  │      │          │  │          SDL_config_os2.h
│  │      │          │  │          SDL_config_symbian.h
│  │      │          │  │          SDL_config_win32.h
│  │      │          │  │          SDL_copying.h
│  │      │          │  │          SDL_cpuinfo.h
│  │      │          │  │          SDL_endian.h
│  │      │          │  │          SDL_error.h
│  │      │          │  │          SDL_events.h
│  │      │          │  │          SDL_getenv.h
│  │      │          │  │          SDL_joystick.h
│  │      │          │  │          SDL_keyboard.h
│  │      │          │  │          SDL_keysym.h
│  │      │          │  │          SDL_loadso.h
│  │      │          │  │          SDL_main.h
│  │      │          │  │          SDL_mouse.h
│  │      │          │  │          SDL_mutex.h
│  │      │          │  │          SDL_name.h
│  │      │          │  │          SDL_opengl.h
│  │      │          │  │          SDL_platform.h
│  │      │          │  │          SDL_quit.h
│  │      │          │  │          SDL_rwops.h
│  │      │          │  │          SDL_stdinc.h
│  │      │          │  │          SDL_syswm.h
│  │      │          │  │          SDL_thread.h
│  │      │          │  │          SDL_timer.h
│  │      │          │  │          SDL_types.h
│  │      │          │  │          SDL_version.h
│  │      │          │  │          SDL_video.h
│  │      │          │  │
│  │      │          │  └─lib
│  │      │          │          dxguid-x64.lib
│  │      │          │          dxguid-x86.lib
│  │      │          │          libfftw-x64-static-mt.lib
│  │      │          │          libfftw-x86-static-mt.lib
│  │      │          │          libjpeg-x64-static-mt.lib
│  │      │          │          libjpeg-x86-static-mt.lib
│  │      │          │          libpng-x64-static-mt.lib
│  │      │          │          libpng-x86-static-mt.lib
│  │      │          │          libtiff-x64-static-mt.lib
│  │      │          │          libtiff-x86-static-mt.lib
│  │      │          │          SDL-x64-static-mt.lib
│  │      │          │          SDL-x86-static-mt.lib
│  │      │          │          zlib-x64-static-mt.lib
│  │      │          │          zlib-x86-static-mt.lib
│  │      │          │
│  │      │          ├─pics
│  │      │          │      belgium.hdr
│  │      │          │      dog1.jpg
│  │      │          │
│  │      │          └─src
│  │      │                  Alignment.cpp
│  │      │                  Alignment.h
│  │      │                  Arithmetic.cpp
│  │      │                  Arithmetic.h
│  │      │                  Calculus.cpp
│  │      │                  Calculus.h
│  │      │                  Color.cpp
│  │      │                  Color.h
│  │      │                  Complex.cpp
│  │      │                  Complex.h
│  │      │                  Control.cpp
│  │      │                  Control.h
│  │      │                  Convolve.cpp
│  │      │                  Convolve.h
│  │      │                  DenseGrid.h
│  │      │                  DFT.cpp
│  │      │                  DFT.h
│  │      │                  Display.cpp
│  │      │                  Display.h
│  │      │                  DisplayWindow.cpp
│  │      │                  DisplayWindow.h
│  │      │                  eigenvectors.h
│  │      │                  Exception.cpp
│  │      │                  Exception.h
│  │      │                  File.cpp
│  │      │                  File.h
│  │      │                  FileCSV.cpp
│  │      │                  FileEXR.cpp
│  │      │                  FileFLO.cpp
│  │      │                  FileHDR.cpp
│  │      │                  FileJPG.cpp
│  │      │                  FileNotImplemented.h
│  │      │                  FilePNG.cpp
│  │      │                  FilePPM.cpp
│  │      │                  FileTGA.cpp
│  │      │                  FileTIFF.cpp
│  │      │                  FileTMP.cpp
│  │      │                  FileWAV.cpp
│  │      │                  Filter.cpp
│  │      │                  Filter.h
│  │      │                  footer.h
│  │      │                  GaussTransform.cpp
│  │      │                  GaussTransform.h
│  │      │                  Geometry.cpp
│  │      │                  Geometry.h
│  │      │                  GKDTree.h
│  │      │                  HDR.cpp
│  │      │                  HDR.h
│  │      │                  header.h
│  │      │                  Image.cpp
│  │      │                  Image.h
│  │      │                  ImageStack.h
│  │      │                  LAHBPCG.cpp
│  │      │                  LAHBPCG.h
│  │      │                  LightField.cpp
│  │      │                  LightField.h
│  │      │                  LinearAlgebra.h
│  │      │                  macros.h
│  │      │                  main.cpp
│  │      │                  main.h
│  │      │                  Network.cpp
│  │      │                  Network.h
│  │      │                  NetworkOps.cpp
│  │      │                  NetworkOps.h
│  │      │                  Operation.cpp
│  │      │                  Operation.h
│  │      │                  Paint.cpp
│  │      │                  Paint.h
│  │      │                  Panorama.cpp
│  │      │                  Panorama.h
│  │      │                  Parser.cpp
│  │      │                  Parser.h
│  │      │                  PatchMatch.cpp
│  │      │                  PatchMatch.h
│  │      │                  Permutohedral.h
│  │      │                  precomputations.py
│  │      │                  Prediction.cpp
│  │      │                  Prediction.h
│  │      │                  Projection.cpp
│  │      │                  Projection.h
│  │      │                  Stack.cpp
│  │      │                  Stack.h
│  │      │                  Statistics.cpp
│  │      │                  Statistics.h
│  │      │                  tables.h
│  │      │                  Wavelet.cpp
│  │      │                  Wavelet.h
│  │      │                  WLS.cpp
│  │      │                  WLS.h
│  │      │
│  │      └─nyu-hooks
│  │              attach_proposals.m
│  │              benchmarkPaths.m
│  │              benchmarkSemantic.m
│  │              cropCamera.m
│  │              cropIt.m
│  │              detectionTasks.m
│  │              getCameraParam.m
│  │              getClassId.m
│  │              getGroundTruth.m
│  │              getGroundTruthBoxes.m
│  │              getImage.m
│  │              getImageSet.m
│  │              getInstSegGT.m
│  │              getMetadata.m
│  │              imdb_eval_nyud2.m
│  │              imdb_from_nyud2.m
│  │              imNameToNum.m
│  │              imNumToName.m
│  │              merge_imdb.m
│  │              putMetadata.m
│  │              README.md
│  │              roidb_from_nyud2.m
│  │              script_data.m
│  │
│  └─wholepose
│      │  demo.py
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
├─ensemble
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
    │  train_parallel.py
    │  train_val_split.mat
    │  T_Pose_model.py
    │
    └─data_process
        │  download_pretrained_model.txt
        │  pose_hrnet.py
        │  wholebody_w48_384x384_adam_lr1e-3.yaml
        │  wholepose_features_extraction.py
        │
        └─config
                default.py
                models.py
                __init__.py