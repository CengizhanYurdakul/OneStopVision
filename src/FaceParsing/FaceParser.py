import cv2
import PIL
import numpy as np
import torchvision.transforms as transforms

from src.FaceParsing.utils import *

class FaceParser:
    def __init__(self) -> None:
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
            
        self.faceParser = BiSeNet(19)
        self.faceParser.load_state_dict(torch.load("src/Models/FaceParsing.pth"))
        self.faceParser.to(self.device)
        self.faceParser.eval()
        
        self.transform = transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                        ])
        
    def alignImage(self, inputImage:np.array, lms68:np.array, outputSize:int=512, transformSize:int=512, xScale:int=1, yScale:int=1, emScale:float=0.1, enablePadding:bool=True):
        src_im = cv2.cvtColor(inputImage, cv2.COLOR_BGR2RGB)
        
        lm_chin = lms68[0: 17]  
        lm_eyebrow_left = lms68[17: 22]  
        lm_eyebrow_right = lms68[22: 27]  
        lm_nose = lms68[27: 31]  
        lm_nostrils = lms68[31: 36]  
        lm_eye_left = lms68[36: 42]  
        lm_eye_right = lms68[42: 48]  
        lm_mouth_outer = lms68[48: 60]  
        lm_mouth_inner = lms68[60: 68]  

        eye_left = np.mean(lm_eye_left, axis=0)
        eye_right = np.mean(lm_eye_right, axis=0)
        eye_avg = (eye_left + eye_right) * 0.5
        eye_to_eye = eye_right - eye_left
        mouth_left = lm_mouth_outer[0]
        mouth_right = lm_mouth_outer[6]
        mouth_avg = (mouth_left + mouth_right) * 0.5
        eye_to_mouth = mouth_avg - eye_avg

        x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
        x /= np.hypot(*x)
        x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
        x *= xScale
        y = np.flipud(x) * [-yScale, yScale]
        c = eye_avg + eye_to_mouth * emScale
        quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
        qsize = np.hypot(*x) * 2
        rsize = None

        img = PIL.Image.fromarray(src_im).convert('RGBA').convert('RGB')

        shrink = int(np.floor(qsize / outputSize * 0.5))
        if shrink > 1:
            rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
            img = img.resize(rsize, PIL.Image.ANTIALIAS)
            quad /= shrink
            qsize /= shrink

        border = max(int(np.rint(qsize * 0.1)), 3)
        crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
                int(np.ceil(max(quad[:, 1]))))
        crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]),
                min(crop[3] + border, img.size[1]))
        if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
            img = img.crop(crop)
            quad -= crop[0:2]

        pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
               int(np.ceil(max(quad[:, 1]))))
        pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0),
               max(pad[3] - img.size[1] + border, 0))
        if enablePadding and max(pad) > border - 4:
            pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
            img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'constant')
            h, w, _ = img.shape
            y, x, _ = np.ogrid[:h, :w, :1]

            blur = qsize * 0.02

            img = np.uint8(np.clip(np.rint(img), 0, 255))

            img = PIL.Image.fromarray(img, 'RGB')
            quad += pad[:2]

        img = PIL.Image.fromarray(np.uint8(img)[:, :, :3])

        img = img.transform((transformSize, transformSize), PIL.Image.QUAD, (quad + 0.5).flatten(),
                            PIL.Image.BILINEAR)

        if outputSize < transformSize:
            img = img.resize((outputSize, outputSize), PIL.Image.ANTIALIAS)

        transformParams = {
            'rsize': rsize,
            'crop': crop,
            'pad': pad,
            'quad': quad + 0.5,
            'new_size': (outputSize, outputSize)
        }

        return img, transformParams
    
    def visualize(self, inputImage:PIL.Image, parseMask:torch.Tensor, stride:int=1):
        im = np.array(inputImage)
        visIm = im.copy().astype(np.uint8)
        visParsingAnno = parseMask.copy().astype(np.uint8)
        visParsingAnno = cv2.resize(visParsingAnno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
        visParsingAnnoColor = np.zeros((visParsingAnno.shape[0], visParsingAnno.shape[1], 3)) + 255
        
        numOfClass = np.max(visParsingAnno)
        
        for pi in range(1, numOfClass + 1):
            index = np.where(visParsingAnno == pi)
            visParsingAnnoColor[index[0], index[1], :] = partColors[pi]

        visParsingAnnoColor = visParsingAnnoColor.astype(np.uint8)
        visIm = cv2.addWeighted(cv2.cvtColor(visIm, cv2.COLOR_RGB2BGR), 0.4, visParsingAnnoColor, 0.6, 0)
        
        return visIm, visParsingAnnoColor
    
    def extractMask(self, inputImage:np.array, lms68:np.array):
        alignedImage, transformParams = self.alignImage(inputImage, lms68)
        transformedImage = self.transform(alignedImage)
        
        with torch.no_grad():
            parseMask = self.faceParser(transformedImage.unsqueeze(0).to(self.device))[0]
        
        parseMask = parseMask.squeeze(0).cpu().numpy().argmax(0)
        
        visImage, maskImage = self.visualize(alignedImage, parseMask)
        
        return np.array(alignedImage), maskImage, visImage