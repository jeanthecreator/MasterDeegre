import numpy as np
import os
from scipy.spatial import distance
from skimage import io, color
from skimage.util import img_as_ubyte
from skimage.color import rgb2lab
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def extract_colors_from_image(image, n_colors=5):
    """
    Extrai as principais cores de uma imagem usando K-Means.

    Args:
        image (np.ndarray): Imagem de entrada (H x W x 3).
        n_colors (int): Número de cores principais a serem extraídas.

    Returns:
        list: Lista de cores principais (RGB).
    """
    # Redimensiona a imagem para um vetor de pixels
    pixels = image.reshape(-1, 3)

    # Aplica K-Means para encontrar as cores principais
    kmeans = KMeans(n_clusters=n_colors, random_state=42)
    kmeans.fit(pixels)
    colors = kmeans.cluster_centers_.astype(int)

    return [list(color) for color in colors]

def map_pixel_colors(image, foreground_image, background_image, n_colors=5):
    """
    Mapeia os pixels de uma imagem para cores do primeiro plano (feijões) e plano de fundo.

    Args:
        image (np.ndarray): Imagem de entrada (H x W x 3).
        foreground_image (np.ndarray): Imagem de exemplo com feijões.
        background_image (np.ndarray): Imagem de exemplo com o fundo.
        n_colors (int): Número de cores a serem extraídas das imagens de exemplo.

    Returns:
        np.ndarray: Imagem processada (H x W) com valores em escala de cinza.
    """
    # Extrair cores principais das imagens de exemplo
    foreground_colors = extract_colors_from_image(foreground_image, n_colors)
    background_colors = extract_colors_from_image(background_image, n_colors)

    # Converter listas de cores para arrays numpy
    F = np.array(foreground_colors)  # Conjunto de cores do primeiro plano
    B = np.array(background_colors)  # Conjunto de cores do plano de fundo

    # Dimensões da imagem
    height, width, _ = image.shape

    # Criar imagem de saída
    output_image = np.zeros((height, width))

    for y in range(height):
        for x in range(width):
            # Cor do pixel atual
            c = image[y, x]

            # Encontrar a cor mais próxima em B
            b_distances = distance.cdist([c], B, metric="euclidean")
            b_min = np.min(b_distances)

            # Encontrar a cor mais próxima em F
            f_distances = distance.cdist([c], F, metric="euclidean")
            f_min = np.min(f_distances)

            # Calcular o valor interpolado
            s = f_min / (f_min + b_min)

            # Mapear o valor para a escala de cinza (0 a 255)
            output_image[y, x] = int(s * 255)

    return output_image

# Exemplo de uso
if __name__ == "__main__":
    # Carregar imagens
    image_path = "C:/Users/jsantos1/OneDrive - QuidelOrtho/Documents/Mestrado/Code/feijao1.png"  # Substitua pelo caminho da imagem principal
    foreground_path = "C:/Users/jsantos1/OneDrive - QuidelOrtho/Documents/Mestrado/Code/cor_feijao.png"  # Substitua pelo caminho da imagem de feijão
    background_path = "C:/Users/jsantos1/OneDrive - QuidelOrtho/Documents/Mestrado/Code/imagem_fundo.png"  # Substitua pelo caminho da imagem de fundo

    input_image = io.imread(image_path)
    input_image_lab = rgb2lab(input_image)
    foreground_image = io.imread(foreground_path)
    background_image = io.imread(background_path)

    # Processar a imagem
    output_image = map_pixel_colors(input_image_lab, foreground_image, background_image, n_colors=6)

    # Normalizar os valores entre 0 e 1
    output_image_normalized = output_image / 255.0

    # Salvar ou visualizar a imagem resultante
    output_folder = "C:/Users/jsantos1/OneDrive - QuidelOrtho/Documents/Mestrado/Code/" 
    os.makedirs(output_folder, exist_ok=True)  # Cria a pasta se não existir
    output_path = os.path.join(output_folder, "bean_output.png")
    io.imsave(output_path, img_as_ubyte(output_image_normalized))

    sample_colors = extract_colors_from_image(background_image)
    # Mostrar cores extraídas
    def display_colors(colors):
        plt.figure(figsize=(12, 2))
        for i, color in enumerate(colors):
            plt.subplot(1, len(colors), i + 1)
            plt.imshow([[color]])
            plt.axis('off')
        plt.show()

    # Mostre as cores extraídas
    display_colors(sample_colors)
