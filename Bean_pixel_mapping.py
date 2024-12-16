import numpy as np
from scipy.spatial import distance
from skimage import io

def map_pixel_colors(image, foreground_colors, background_colors):
    """
    Mapeia os pixels de uma imagem para cores do primeiro plano (feijões) e plano de fundo.

    Args:
        image (np.ndarray): Imagem de entrada (H x W x 3).
        foreground_colors (list): Lista de cores do primeiro plano (F).
        background_colors (list): Lista de cores do plano de fundo (B).

    Returns:
        np.ndarray: Imagem processada (H x W) com valores em escala de cinza.
    """
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
    # Carregar imagem (Fig. 2a)
    image_path = "path/to/input/image.jpg"  # Substitua pelo caminho da imagem
    input_image = io.imread(image_path)

    # Definir cores do primeiro plano (feijões) e plano de fundo manualmente
    foreground_colors = [
        [120, 80, 50],  # Exemplo de cores do feijão
        [130, 90, 60],
        # Adicione mais cores aqui
    ]

    background_colors = [
        [200, 200, 200],  # Exemplo de cores de fundo
        [220, 220, 220],
        # Adicione mais cores aqui
    ]

    # Processar a imagem
    output_image = map_pixel_colors(input_image, foreground_colors, background_colors)

    # Salvar ou visualizar a imagem resultante
    output_path = "path/to/output/image.jpg"  # Substitua pelo caminho de saída
    io.imsave(output_path, output_image.astype(np.uint8))
