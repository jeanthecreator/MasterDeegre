import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def apply_lasca(image_path):
    # Carregar a imagem
    image = cv2.imread(image_path)
    
    # Definir tamanho da janela 2x2
    window_size = 2
    h, w, _ = image.shape
    
    # Criar matriz para armazenar o resultado LASCA
    lasca_result = np.zeros((h // window_size, w // window_size, 3), dtype=np.uint8)
    
    # Aplicar LASCA na imagem percorrendo blocos 2x2 em cada canal de cor
    for i in range(0, h - window_size + 1, window_size):
        for j in range(0, w - window_size + 1, window_size):
            block = image[i:i + window_size, j:j + window_size]
            mean_intensity = np.mean(block, axis=(0, 1))
            std_dev = np.std(block, axis=(0, 1))
            
            # Evitar divisão por zero
            contrast = np.where(mean_intensity == 0, 0, std_dev / mean_intensity)
            
            lasca_result[i // window_size, j // window_size] = (contrast * 255).astype(np.uint8)
    
    # Identificar cores presentes na imagem do LASCA
    unique_colors = Counter(map(tuple, lasca_result.reshape(-1, 3)))
    
    return lasca_result, unique_colors

def filter_lasca_image(lasca_image, colors, threshold=100, brightness_threshold=30):
    h, w, _ = lasca_image.shape
    filtered_image = np.copy(lasca_image)
    
    for i in range(h):
        for j in range(w):
            color = tuple(lasca_image[i, j])
            brightness = np.mean(color)
            
            # Remover cores com mais de 100 ocorrências, brilho menor que 30 ou qualquer tom de azul
            if colors[color] >= threshold or brightness < brightness_threshold or color[2] > color[0] or color[2] > color[1]:
                filtered_image[i, j] = [255, 255, 255]  # Substituir por branco
    
    return filtered_image

def connect_pixels(image):
    h, w, _ = image.shape
    connected_image = np.copy(image)
    
    for i in range(h):
        for j in range(w):
            if np.all(image[i, j] == [255, 255, 255]):  # Se for branco, pula
                continue
            
            # Procurar o próximo pixel de cor diferente de branco na horizontal
            for nj in range(j + 1, min(j + 6, w)):
                if not np.all(image[i, nj] == [255, 255, 255]):
                    cv2.line(connected_image, (j, i), (nj, i), (255, 0, 0), 1)
                    break
            
            # Procurar o próximo pixel de cor diferente de branco na vertical
            for ni in range(i + 1, min(i + 6, h)):
                if not np.all(image[ni, j] == [255, 255, 255]):
                    cv2.line(connected_image, (j, i), (j, ni), (255, 0, 0), 1)
                    break
    
    return connected_image

# Caminho da imagem
image_path = "C:/Users/jsantos1/OneDrive - QuidelOrtho/Documents/Mestrado/Code/Images/1_speckle.bmp"

# Aplicar LASCA
lasca_image, colors = apply_lasca(image_path)

# Aplicar filtro para remover cores indesejadas
filtered_lasca_image = filter_lasca_image(lasca_image, colors, threshold=100, brightness_threshold=30)

# Conectar pixels com vermelho
connected_lasca_image = connect_pixels(filtered_lasca_image)

# Mostrar imagem final
plt.imshow(cv2.cvtColor(connected_lasca_image, cv2.COLOR_BGR2RGB))
plt.title("Imagem LASCA com Conexões Vermelhas")
plt.axis("off")
plt.show()

# Exibir as cores encontradas
print("Cores únicas na imagem resultante após filtragem e conexão:")
for color, count in colors.items():
    brightness = np.mean(color)
    if count < 100 and brightness >= 30 and color[2] <= color[0] and color[2] <= color[1]:
        print(f"Cor: {color}, Quantidade: {count}, Brilho: {brightness}")
