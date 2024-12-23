import cv2
import numpy as np
import os
from skimage import io, color
from skimage.util import img_as_ubyte
import cv2
import numpy as np
from sklearn.cluster import KMeans

def get_dominant_colors(image_path, n_colors=6):
    """
    Extrair as cores dominantes de uma imagem usando K-Means.
    
    :param image_path: Caminho da imagem.
    :param n_colors: Número de cores dominantes a serem extraídas.
    :return: Lista de cores dominantes (em RGB).
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Converter para RGB
    pixels = image.reshape(-1, 3)  # Reformatar para lista de pixels
    
    # Usar K-Means para encontrar as cores dominantes
    kmeans = KMeans(n_clusters=n_colors, random_state=42).fit(pixels)
    return kmeans.cluster_centers_.astype(int)  # Retornar as cores dominantes


def classify_pixels(image_path, bean_colors, bg_colors):
    """
    Classificar pixels da imagem como grãos ou fundo com base nas cores de referência.
    
    :param image_path: Caminho da imagem original.
    :param bean_colors: Lista de cores dominantes dos grãos.
    :param bg_colors: Lista de cores dominantes do fundo.
    :return: Imagem binária com grãos em preto e fundo em branco.
    """
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Converter para RGB
    height, width, _ = image.shape
    
    # Inicializar a imagem de saída binária
    output_mask = np.zeros((height, width), dtype=np.uint8)
    
    # Iterar sobre os pixels para classificação
    for i in range(height):
        for j in range(width):
            pixel = image_rgb[i, j]
            
            # Calcular distâncias até as cores dos grãos e do fundo
            dist_to_beans = [np.linalg.norm(pixel - color) for color in bean_colors]
            dist_to_bg = [np.linalg.norm(pixel - color) for color in bg_colors]
            
            # Classificar o pixel
            if min(dist_to_beans) < min(dist_to_bg):
                output_mask[i, j] = 0  # Preto: Grãos
            else:
                output_mask[i, j] = 255  # Branco: Fundo
    
    return output_mask


# Caminhos das imagens
input_image = "C:/Users/jsantos1/OneDrive - QuidelOrtho/Documents/Mestrado/Code/Images/feijao1.png"
bean_reference = "C:/Users/jsantos1/OneDrive - QuidelOrtho/Documents/Mestrado/Code/Images/10_BRANCO_Zoom_40.bmp"
background_reference = "C:/Users/jsantos1/OneDrive - QuidelOrtho/Documents/Mestrado/Code/Images/Imagem_fundo_PB.png"

# Obter as cores dominantes
bean_colors = get_dominant_colors(bean_reference, n_colors=6)
bg_colors = get_dominant_colors(background_reference, n_colors=4)

# Classificar os pixels e gerar a máscara binária
output_mask = classify_pixels(input_image, bean_colors, bg_colors)
output_image_normalized = output_mask / 255.0

# Salvar a saída
output_path = "C:/Users/jsantos1/OneDrive - QuidelOrtho/Documents/Mestrado/Code/Images/"
os.makedirs(output_path, exist_ok=True)  # Cria a pasta se não existir
output_path = os.path.join(output_path, "bean_output_new.png")
io.imsave(output_path, img_as_ubyte(output_image_normalized))





