import torch
from rmn import RMN

from src.FacialAnalysis.utils import *
from src.FacialAnalysis.AgeGenderEstimator import AgeGenderEstimator

class FaceAnalyzer:
    def __init__(self) -> None:
        
        if torch.cuda.is_available:
            self.device = "cuda"
        else:
            self.device = "cpu"
            
        self.ageGenderEstimator = AgeGenderEstimator(
            "src/Models/model_utk_age_gender_4.23_97.69.pth.tar",
            self.device,
            half=False,
            use_persons=False,
            disable_faces=False,
            verbose=False
        )
        
        self.emotionRecognizer = RMN(face_detector=False)
        
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        
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
    
    def cropFace(self, inputImage:np.array, boxes:np.array) -> np.array:
        box = boxes[:4]
        croppedFace = inputImage[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
        return croppedFace
        
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
    
    def preprocess(self, img:np.array) -> torch.Tensor:
        img = letterbox(im=img, newShape=(224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = img / 255.0
        img = (img - self.mean) / self.std
        img = img.astype(dtype=np.float32)

        img = img.transpose((2, 0, 1))
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img)
        img = img.unsqueeze(0)
        return img
    
    def postprocess(self, predicted:torch.Tensor) -> dict:
        ageOutput = predicted[:, 2]
        genderOutput = predicted[:, :2].softmax(-1)
        genderProbs, genderIndx = genderOutput.topk(1)
        
        age = ageOutput[0].item()
        age = age * (self.ageGenderEstimator.meta.max_age - self.ageGenderEstimator.meta.min_age) + self.ageGenderEstimator.meta.avg_age
        age = round(age, 2)
        
        gender = "Male" if genderIndx[0].item() == 0 else "Female"
        genderScore = genderProbs[0].item()
        
        return gender, round(genderScore*100, 2), age
        
    def predictAgeGender(self, inputImage:np.array, boxes:np.array) -> dict:
        index = self.findBiggestFace(boxes)
        croppedFace = self.cropFace(inputImage, boxes[0][index])
        inputTensor = self.preprocess(croppedFace)
        
        with torch.no_grad():
            predicted = self.ageGenderEstimator.inference(inputTensor.to(self.device))
            
        gender, genderScore, age = self.postprocess(predicted)
        
        return gender, genderScore, age
    
    def predictEmotion(self, inputImage:np.array, boxes:np.array):
        index = self.findBiggestFace(boxes)
        squareBox = self.convertBox2Square(boxes[0][index][0], boxes[0][index][1], boxes[0][index][2], boxes[0][index][3])
        squareBox.append(boxes[0][index][0])
        croppedFace = self.cropFace(inputImage, np.array(squareBox))
        
        with torch.no_grad():
            emotion, emotionScore, _ = self.emotionRecognizer.detect_emotion_for_single_face_image(croppedFace)
            
        return emotion, round(emotionScore*100, 2)
    
    def main(self, inputImage:np.array, boxes:np.array) -> dict:
        self.emotionRecognizer
        gender, genderScore, age = self.predictAgeGender(inputImage, boxes)
        emotion, emotionScore = self.predictEmotion(inputImage, boxes)
        return gender, genderScore, age, emotion, emotionScore