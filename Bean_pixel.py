import cv2
import numpy as np
import os
from sklearn.cluster import KMeans
from skimage import io
from skimage.util import img_as_ubyte

def get_dominant_colors_from_folder(folder_path, file_prefix, n_colors=3):
    """
    Extrair cores dominantes de várias imagens em uma pasta.
    
    :param folder_path: Caminho para a pasta das imagens.
    :param file_prefix: Prefixo dos arquivos (ex.: "fundo_", "grao_").
    :param n_colors: Número de cores dominantes a serem extraídas por imagem.
    :return: Lista acumulada de cores dominantes (em RGB).
    """
    all_colors = []
    for file_name in sorted(os.listdir(folder_path)):
        if file_name.startswith(file_prefix) and file_name.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, file_name)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Converter para RGB
            pixels = image.reshape(-1, 3)  # Reformatar para lista de pixels
            
            # Usar K-Means para encontrar as cores dominantes
            kmeans = KMeans(n_clusters=n_colors, random_state=42).fit(pixels)
            all_colors.extend(kmeans.cluster_centers_.astype(int))  # Acumular as cores dominantes
    
    return np.array(all_colors)  # Retornar todas as cores acumuladas


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


# Caminhos das pastas de referência
bean_folder = "C:/Users/jsantos1/OneDrive - QuidelOrtho/Documents/Mestrado/Code/Images_comp_bean/"  # Pasta contendo grao_001, grao_002, ...
bg_folder = "C:/Users/jsantos1/OneDrive - QuidelOrtho/Documents/Mestrado/Code/Images_comp_bg/"  # Pasta contendo fundo_01, fundo_02, ...

# Obter as cores dominantes de todas as imagens de referência
bean_colors = get_dominant_colors_from_folder(bean_folder, file_prefix="grao_", n_colors=3)
bg_colors = get_dominant_colors_from_folder(bg_folder, file_prefix="fundo_", n_colors=3)

# Caminho da imagem original
input_image = "C:/Users/jsantos1/OneDrive - QuidelOrtho/Documents/Mestrado/Code/Images/feijao1.png"

# Classificar os pixels e gerar a máscara binária
output_mask = classify_pixels(input_image, bean_colors, bg_colors)
output_image_normalized = output_mask / 255.0

# Salvar a saída
output_path = "C:/Users/jsantos1/OneDrive - QuidelOrtho/Documents/Mestrado/Code/Images/"
os.makedirs(output_path, exist_ok=True)  # Cria a pasta se não existir
output_path = os.path.join(output_path, "bean_output_new_def.png")
io.imsave(output_path, img_as_ubyte(output_image_normalized))