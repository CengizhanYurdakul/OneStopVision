import cv2
import numpy as np

def letterbox(im:np.array, newShape:tuple=(640, 640), color:tuple=(0, 0, 0), scaleup:bool=True) -> np.array:
    shape = im.shape[:2]
    if isinstance(newShape, int):
        newShape = (newShape, newShape)

    if im.shape[0] == newShape[0] and im.shape[1] == newShape[1]:
        return im

    r = min(newShape[0] / shape[0], newShape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = newShape[1] - new_unpad[0], newShape[0] - new_unpad[1] 

    dw /= 2 
    dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color) 
    return im