import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# Rotacionar imagem
def rotate_image(image, angle, scale=1.0):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

# Carregar imagem sem conversão de cores
def load_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Erro: O arquivo '{image_path}' não foi encontrado.")
    
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # Mantém a imagem original em BGR
    if image is None:
        raise ValueError(f"Erro: Não foi possível carregar a imagem '{image_path}'. Verifique o caminho e a integridade do arquivo.")
    
    return image

# Aplicação do LAsCA (Contraste Espacial do Laser Speckle) na imagem original com redução de tamanho
def apply_lasca(image, window_size=2, output_path="lasca_result.png"):
    h, w, c = image.shape
    new_h, new_w = h // window_size, w // window_size
    lasca_image = np.zeros((new_h, new_w, c), dtype=np.float32)
    
    for y in range(0, new_h):
        for x in range(0, new_w):
            for channel in range(c):
                region = image[y*window_size:(y+1)*window_size, x*window_size:(x+1)*window_size, channel].astype(np.float32)
                mean_intensity = np.mean(region)
                std_intensity = np.std(region)
                
                if mean_intensity > 0:
                    contrast = std_intensity / mean_intensity  # Aplicando a equação do LAsCA
                else:
                    contrast = 0
                
                lasca_image[y, x, channel] = contrast * 255
    
    lasca_image = np.clip(lasca_image, 0, 255).astype(np.uint8)
    cv2.imwrite(output_path, lasca_image)
    print(f"Imagem LAsCA salva em: {output_path}")
    return lasca_image

# Criar máscara para destacar todos os tons de verde
def apply_green_mask(image_path, output_mask_path="green_mask.png"):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Erro ao carregar a imagem LAsCA: {image_path}")
    
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([20, 40, 40])  # Faixa ajustada para capturar todos os tons de verde
    upper_green = np.array([100, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Aplicar operações morfológicas para suavizar e melhorar a detecção
    kernel = np.ones((9, 9), np.uint8)  # Kernel ajustado para suavização
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    cv2.imwrite(output_mask_path, mask)
    print(f"Máscara de Verde salva em: {output_mask_path}")
    return mask

# Pipeline - Etapa 1: Gerar LAsCA e salvar
image_path = "C:/Users/jsantos1/OneDrive - QuidelOrtho/Documents/Mestrado/Code/Images/1_speckle.bmp"  # Substitua pela imagem real
lasca_output = "lasca_result.png"
mask_output = "green_mask.png"

try:
    image = load_image(image_path)
    lasca_result = apply_lasca(image, window_size=2, output_path=lasca_output)
    print("LAsCA gerado. Agora execute o segundo script para gerar a máscara.")
except Exception as e:
    print(e)
