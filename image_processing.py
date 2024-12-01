from pathlib import Path
import cv2
import numpy as np

def image_mask(image, threshold_value=50):

    _, thres_hold = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
    contornos, _ = cv2.findContours(thres_hold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mascara = np.zeros_like(image)

    if contornos:
        contorno_principal = max(contornos, key=cv2.contourArea)
        cv2.drawContours(mascara, [contorno_principal], -1, 255, thickness=cv2.FILLED)

        return mascara

def apply_mask(image, mascara):

    return cv2.bitwise_and(image, image, mask=mascara)