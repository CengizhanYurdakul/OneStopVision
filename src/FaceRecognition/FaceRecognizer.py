import cv2
import numpy as np
from skimage import transform as trans

from src.FaceRecognition.Arcface import *

class FaceRecognizer:
    def __init__(self) -> None:
        self.arcface = IResNet()
        self.arcface.load_state_dict(torch.load("src/Models/ArcfaceR100.pth"))
        self.arcface.eval()
        
        self.arcfaceTemplate = np.array([[38.2946, 51.6963],
                                         [73.5318, 51.5014],
                                         [56.0252, 71.7366],
                                         [41.5493, 92.3655],
                                         [70.7299, 92.2041]], dtype=np.float32)
        
        self.similarityTransform = trans.SimilarityTransform()
        
    def alignImage(self, image:np.array, lms:np.array) -> np.array:
        lms5 = lms
        
        self.similarityTransform.estimate(lms5, self.arcfaceTemplate)
        transformMatrix = self.similarityTransform.params[0:2, :]
        
        alignedImage = cv2.warpAffine(image, transformMatrix, (112, 112)).astype("uint8")
        return alignedImage
    
    def preprocess(self, inputImage:np.array) -> torch.Tensor:
        inputImage = np.transpose(inputImage, (2, 0, 1))
        inputImage = torch.from_numpy(inputImage).unsqueeze(0).float()
        inputImage.div_(255).sub_(0.5).div_(0.5)
        return inputImage
    
    def l2Norm(self, input:torch.Tensor, axis:int=1) -> torch.Tensor:
        norm = torch.norm(input,2,axis,True)
        output = torch.div(input, norm)
        return output
    
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
            
    def compare(self, image1:list, image2:list) -> float:
        image1Array, image1Box, image1Lms = image1[0], image1[1], image1[2]
        image2Array, image2Box, image2Lms = image2[0], image2[1], image2[2]
        
        index1 = self.findBiggestFace(image1Box)
        index2 = self.findBiggestFace(image2Box)
        
        alignedImage1 = self.alignImage(image1Array, image1Lms[0][index1])
        alignedImage2 = self.alignImage(image2Array, image2Lms[0][index2])
        
        input1 = self.preprocess(alignedImage1)
        input2 = self.preprocess(alignedImage2)
        
        with torch.no_grad():
            identity1 = self.arcface(input1)
            identity2 = self.arcface(input2)
        
        normalizedIdentity1 = self.l2Norm(identity1)
        normalizedIdentity2 = self.l2Norm(identity2)
        
        cosineSimilarity = torch.nn.functional.cosine_similarity(normalizedIdentity1, normalizedIdentity2)
        return cosineSimilarity, alignedImage1, alignedImage2, identity1, identity2