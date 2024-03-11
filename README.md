# OneStopVision
Unleash the power of computer vision in your projects!  OneStopVision is your one-stop shop for pre-trained algorithms, offering a comprehensive toolkit for facial analysis, object detection, and depth estimation. Dive into tasks like face recognition, landmark extraction, and head pose estimation – all readily available and accompanied by a user-friendly README for smooth installation and integration.

## Features at a Glance
| Feature | Description |
| ------- | ----------- |
| Face Detection | OneStopVision detects human faces, returning bounding boxes and key landmarks in a convenient JSON format. |
| Face Recognition | OneStopVision calculates inter-face cosine similarities for recognition, extracting unique identity features in a single step. |
| Facial Attribute Analysis | OneStopVision analyzes age, gender, and emotion with probabilistic insights, unlocking a deeper understanding of your visual content. |
| Face Parsing | OneStopVision isolates facial regions with intelligent parsing, generating detailed masks for unparalleled control over your visual data. |
| Landmark Extraction | OneStopVision extracts 68 key landmarks and delivers them in convenient JSON format. |
| Head Pose Estimation | OneStopVision estimates head pose in yaw, pitch, and roll, even visualizing it directly on the image for an intuitive understanding of facial orientation. |
| ControlNet Operations | OneStopVision empowers you with M-LSD, HED, OpenPose, depth estimation, and semantic segmentation, all visualized for seamless integration and groundbreaking visual analysis. |

If you want to see how the features work, visit [README_VIS.md](README_VIS.md).

## Install
```
conda create --name onestop python==3.9.18 -y
conda activate onestop

# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

bash downloadModels.sh
```

## Run
```
steamlit run app.py
```

## Acknowledgement
- **Face Detection:** The structure used for face detection is taken from the [DSFD-Pytorch-Inference repository](https://github.com/hukkelas/DSFD-Pytorch-Inference).
- **Face Recognition:** The model used for face recognition was taken from the [insightface repository](https://github.com/deepinsight/insightface). [Arcface R100](https://arxiv.org/abs/1801.07698) backbone was used to compare the identities.
- **Facial Attribute Analysis:** [MiVOLO](https://github.com/wildchlamydia/mivolo) repository was used for Gender and age estimation. [ResidualMaskingNetwork repository](https://github.com/phamquiluan/ResidualMaskingNetwork) was used for emotion recognition.
- **Face Parsing:** The [face-parsing.PyTorch repository](https://github.com/zllrunning/face-parsing.PyTorch) was used for face parsing.
- **Landmark Extraction:** The [face-alignment](https://github.com/1adrianb/face-alignment) repository was used for 68 landmark extraction.
- **Head Pose Estimation:** The [6DRepNet](https://github.com/thohemp/6DRepNet) repository was used for head pose estimation.
- **ControlNet Operations:** The [controlnet-aux library](https://github.com/huggingface/controlnet_aux), [lllyasviel/ControlNet](https://huggingface.co/lllyasviel/ControlNet) models were used for ControlNet operations.