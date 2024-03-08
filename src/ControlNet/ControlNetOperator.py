import cv2
import torch
from PIL import Image

from src.ControlNet.utils import *

class ControlNetOperator:
    def __init__(self) -> None:
        self.mlsdDetector = None
        self.hedDetector = None
        self.openposeDetector = None
        self.depthEstimator = None
        self.semanticProcessor = None
        self.semanticSegmentator = None
        
    
    def mainCanny(self, inputImage:np.array):
        image = cv2.Canny(inputImage, 100, 200)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        return image
    
    def mainMLSD(self, inputImage):
        if self.mlsdDetector is None:
            from controlnet_aux import MLSDdetector
            self.mlsdDetector = MLSDdetector.from_pretrained('lllyasviel/ControlNet')
            
        mlsdImage = self.mlsdDetector(Image.fromarray(inputImage))
        return np.array(mlsdImage)
    
    def mainHED(self, inputImage):
        if self.hedDetector is None:
            from controlnet_aux import HEDdetector
            self.hedDetector = HEDdetector.from_pretrained('lllyasviel/ControlNet')
            
        hedImage = self.hedDetector(Image.fromarray(inputImage))
        return np.array(hedImage)
    
    def mainOpenPose(self, inputImage):
        if self.openposeDetector is None:
            from controlnet_aux import OpenposeDetector
            self.openposeDetector = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')
            
        openposeImage = self.openposeDetector(inputImage)
        return np.array(openposeImage)
    
    def mainDepth(self, inputImage):
        if self.depthEstimator is None:
            from transformers import pipeline
            self.depthEstimator = pipeline("depth-estimation")
            
        image = self.depthEstimator(Image.fromarray(inputImage))['depth']
        image = np.array(image)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        return image
            
    def mainSemanticSegmentation(self, inputImage):
        if self.semanticSegmentator is None:
            from transformers import AutoImageProcessor, UperNetForSemanticSegmentation
            self.semanticProcessor = AutoImageProcessor.from_pretrained("openmmlab/upernet-convnext-small")
            self.semanticSegmentator = UperNetForSemanticSegmentation.from_pretrained("openmmlab/upernet-convnext-small")
            
        inputImage_PIL = Image.fromarray(inputImage)
        pixelValues = self.semanticProcessor(inputImage_PIL, return_tensors="pt").pixel_values

        with torch.no_grad():
            outputs = self.semanticSegmentator(pixelValues)

        seg = self.semanticProcessor.post_process_semantic_segmentation(outputs, target_sizes=[inputImage_PIL.size[::-1]])[0]
        colorSeg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8) # height, width, 3

        for label, color in enumerate(palette):
            colorSeg[seg == label, :] = color

        colorSeg = colorSeg.astype(np.uint8)
        
        return colorSeg