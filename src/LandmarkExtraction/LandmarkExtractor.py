import cv2
import torch
import numpy as np
import face_alignment

class LandmarkExtractor:
    def __init__(self) -> None:
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
            
        self.landmarkExtractor = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device=device)
    
    def findBiggestFace(self, boxes:np.array) -> int:
        biggestIndex = 0
        biggestArea = 0
        box = boxes[0]
        for i in range(box.shape[0]):
            area = (box[i][2] - box[i][0]) * (box[i][3] - box[i][1])
            if area > biggestArea:
                biggestArea = area
                biggestIndex = i
        return biggestIndex
    
    def parseLandmarkExtractionOutputs(self, inputImage:np.array, landmark68:np.array):
        outDict = {
            "Lms68": [[i[0], i[1]] for i in landmark68]
        }
        outImage = inputImage.copy()
        
        for i in landmark68:
            outImage = cv2.circle(outImage, (int(i[0]), int(i[1])), 3, (0, 0, 255), -1)
            
        return outImage, outDict
            
        
    
    def main(self, inputImage:np.array, boxes:np.array) -> np.array:
        index = self.findBiggestFace(boxes)
        landmark68 = self.landmarkExtractor.get_landmarks(inputImage, detected_faces=[boxes[0][index][:4]])[0]
        return landmark68
    