

import cv2 as cv
import numpy as np
from paddleocr import TextDetection


# PaddleOCR model method
model = TextDetection()

def text_detection_pocr(img:object):
    """Text detection function using PaddleOCR

    Args:
        img (object): image

    Returns:
        list: contours of detected text regions
    """
    
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    res = model.predict(img_rgb)

    dt_polys = res[0]['dt_polys']
    scores = res[0]['dt_scores'] #
    contours = [np.array(poly, dtype=np.int32) for poly in dt_polys]


    return contours, scores



