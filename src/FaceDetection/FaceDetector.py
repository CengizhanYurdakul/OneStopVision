import cv2
import numpy as np
import face_detection

class FaceDetector:
    def __init__(self) -> None:
        self.faceDetector = face_detection.build_detector("RetinaNetResNet50", confidence_threshold=.3, nms_iou_threshold=.3, max_resolution=1920)
    
    def parseFaceDetectionOutputs(self, image:np.array, boxes:np.array, landmarks:np.array) -> dict:
        boxArray, lmsArray = boxes[0], landmarks[0]
        numDetectedFace = boxArray.shape[0]
        
        outDict = {}
        outImage = image.copy()
        
        for i in range(numDetectedFace):
            box = boxArray[i][:4]
            lms = lmsArray[i]
            
            outImage = cv2.rectangle(outImage, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 3)
            outImage = cv2.circle(outImage, (int(lms[0][0]), int(lms[0][1])), 3, (0, 0, 255), -1)
            outImage = cv2.circle(outImage, (int(lms[1][0]), int(lms[1][1])), 3, (0, 0, 255), -1)
            outImage = cv2.circle(outImage, (int(lms[2][0]), int(lms[2][1])), 3, (0, 0, 255), -1)
            outImage = cv2.circle(outImage, (int(lms[3][0]), int(lms[3][1])), 3, (0, 0, 255), -1)
            outImage = cv2.circle(outImage, (int(lms[4][0]), int(lms[4][1])), 3, (0, 0, 255), -1)
            
            outDict[i] = {
                "Box": {"x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3]},
                "Lms": {
                    "LeftEye": [lms[0][0], lms[0][1]],
                    "RightEye": [lms[1][0], lms[1][1]],
                    "Nose": [lms[2][0], lms[2][1]],
                    "LeftMouth": [lms[3][0], lms[3][1]],
                    "RightMouth": [lms[4][0], lms[4][1]],
                },
                "Confidence": boxArray[i][4]
            }
        
        return outImage, outDict
    
    def detect(self, inputImage:np.array):
        boxes, landmarks = self.faceDetector.batched_detect_with_landmarks(inputImage[np.newaxis, ...])
        return boxes, landmarks