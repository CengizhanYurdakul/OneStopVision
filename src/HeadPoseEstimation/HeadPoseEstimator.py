import numpy as np
from PIL import Image
from sixdrepnet import SixDRepNet
from torchvision import transforms

from src.HeadPoseEstimation.utils import *

class HeadPoseEstimator:
    def __init__(self) -> None:
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
            
        self.poseEstimator = SixDRepNet(dict_path="src/Models/6DRepNet_300W_LP_AFLW2000.pth")
        
        self.transforms = transforms.Compose([transforms.Resize(224),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        
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
    
    def convertBox2Square(self, xmin:int, ymin:int, xmax:int, ymax:int):
        centerX = (xmin + xmax) // 2
        centerY = (ymin + ymax) // 2

        squareLength = ((xmax - xmin) + (ymax - ymin)) // 2 // 2
        squareLength *= 1.1

        xmin = int(centerX - squareLength)
        ymin = int(centerY - squareLength)
        xmax = int(centerX + squareLength)
        ymax = int(centerY + squareLength)
        return [xmin, ymin, xmax, ymax]
        
    def preprocess(self, inputImage:np.array, boxes:np.array):
        xMin = int(boxes[0])
        yMin = int(boxes[1])
        xMax = int(boxes[2])
        yMax = int(boxes[3])
        xMin, yMin, xMax, yMax = self.convertBox2Square(xMin, yMin, xMax, yMax)
        self.bboxW = abs(xMax - xMin)
        self.bboxH = abs(yMax - yMin)

        self.xMin = max(0, xMin-int(0.2*self.bboxH))
        self.yMin = max(0, yMin-int(0.2*self.bboxW))
        self.xMax = xMax+int(0.2*self.bboxH)
        self.yMax = yMax+int(0.2*self.bboxW)
        
        croppedImage = inputImage[yMin:yMax, xMin:xMax]
        
        return croppedImage
    
    def main(self, inputImage:np.array, boxes:np.array):
        index = self.findBiggestFace(boxes)
        
        croppedImage = self.preprocess(inputImage, boxes[0][index])
        
        with torch.no_grad():
            pitchDegree, yawDegree, rollDegree = self.poseEstimator.predict(croppedImage)
            
        visImage = self.poseEstimator.draw_axis(inputImage, yawDegree, pitchDegree, rollDegree, self.xMin + int(.5*(
                    self.xMax-self.xMin)), self.yMin + int(.5*(self.yMax-self.yMin)), size=self.bboxW)
        
        
        return visImage, yawDegree[0], pitchDegree[0], rollDegree[0]