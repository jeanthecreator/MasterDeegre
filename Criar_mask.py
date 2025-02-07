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
def apply_green_mask(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Exibir histogramas dos valores HSV
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.hist(hsv[:, :, 0].flatten(), bins=50, color='green', alpha=0.7)
    plt.title("Histograma de Hue (Matiz)")
    plt.subplot(1, 3, 2)
    plt.hist(hsv[:, :, 1].flatten(), bins=50, color='blue', alpha=0.7)
    plt.title("Histograma de Saturação")
    plt.subplot(1, 3, 3)
    plt.hist(hsv[:, :, 2].flatten(), bins=50, color='gray', alpha=0.7)
    plt.title("Histograma de Valor (Brilho)")
    plt.show()
    
    # Ajustar faixa com base nos histogramas
    lower_green = np.array([30, 50, 20])  # Faixa mais ampla para verde
    upper_green = np.array([100, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Aplicar operações morfológicas para suavizar e melhorar a detecção
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    return mask

# Pipeline - Gerar LAsCA e máscara
image_path = "C:/Users/jsantos1/OneDrive - QuidelOrtho/Documents/Mestrado/Code/Images/1_speckle.bmp"  # Substitua pela imagem real
lasca_output = "lasca_result.png"
mask_output = "green_mask.png"

try:
    image = load_image(image_path)
    lasca_result = apply_lasca(image, window_size=2, output_path=lasca_output)
    green_mask = apply_green_mask(lasca_result)
    
    # Exibir as cores detectadas
    hsv = cv2.cvtColor(lasca_result, cv2.COLOR_BGR2HSV)
    green_pixels = hsv[green_mask > 0]
    
    if green_pixels.size > 0:
        print(f"Faixa de cores identificadas na imagem LAsCA:")
        print(f"Hue (Matiz): {np.min(green_pixels[:, 0])} - {np.max(green_pixels[:, 0])}")
        print(f"Saturação: {np.min(green_pixels[:, 1])} - {np.max(green_pixels[:, 1])}")
        print(f"Brilho: {np.min(green_pixels[:, 2])} - {np.max(green_pixels[:, 2])}")
    else:
        print("Nenhum tom de verde detectado na imagem LAsCA.")
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(lasca_result, cv2.COLOR_BGR2RGB))
    plt.title("Imagem LAsCA")
    plt.subplot(1, 2, 2)
    plt.imshow(green_mask, cmap='gray')
    plt.title("Máscara de Verde Detectado")
    plt.show()

except Exception as e:
    print(e)
