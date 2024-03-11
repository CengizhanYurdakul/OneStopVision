# TODO
## V2
- [] Different models will be implemented for **face detection**. For example [SCRFD](https://github.com/deepinsight/insightface/tree/master/detection/scrfd) models from insightface etc.
- [] Multiple image and video processing for **face detection**, followed by saving output files (.JSON or .npy) for each of the processed images and videos will be implemented.
- [] New **face recognition** models will be implemented. For example [FaceNet](https://github.com/davidsandberg/facenet), [GhostFaceNets](https://github.com/HamadYA/GhostFaceNets), [AdaFace](https://github.com/mk-minchul/adaface) etc.
- [] In addition to the current Arcface R100 recognition model, models trained with [different backbone and different datasets](https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch#1-training-on-single-host-gpu) will be implemented.
- [] Multiple image option will be implemented for **facial attribute analysis**. Output will be availabe for download format (.csv or .pkl).
- [] A new model will be implemented for **face parsing** which is high quality and segments only the face. The output mask will be aligned to the original frame and given in the same format.
- [] A structure that receives video input for **landmark extraction** will be implemented. Here, a high quality model that uses tracking algorithms and minimises flicker will be implemented.
- [] In **ControlNet operations**, a structure that receives multiple image and video inputs and records the outputs frame-by-frame will be implemented. 