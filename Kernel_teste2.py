import numpy as np
import os
import cv2
import math
import matplotlib.pyplot as plt

def generate_ellipse_kernels(sizes, eccentricities, angles, image_size=(200, 200), output_folder="kernels"):
    """
    Gera kernels de elipses baseados em tamanhos, excentricidades e ângulos.
    :param sizes: Lista de tamanhos (pequeno, normal, grande).
    :param eccentricities: Lista de fatores de excentricidade.
    :param angles: Lista de ângulos para rotação.
    :param image_size: Tamanho da imagem para o kernel.
    :return: Lista de imagens com kernels.
    """
    kernels = []
    size_labels = ["small", "normal", "large"]
    ecc_labels = ["low_ecc", "med_ecc", "high_ecc"]

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for size_idx, size in enumerate(sizes):
        for ecc_idx, ecc in enumerate(eccentricities):
            for angle in angles:
                # Criar imagem em branco
                kernel = np.zeros(image_size, dtype=np.uint8)
                center = (image_size[0] // 2, image_size[1] // 2)
                
                # Definir dimensões da elipse
                major_axis = int(size * (1 + ecc))  # Eixo maior ajustado pela excentricidade
                minor_axis = int(size / (1 + ecc))  # Eixo menor ajustado
                
                # Desenhar elipse
                cv2.ellipse(kernel, center, (major_axis, minor_axis), angle, 0, 360, 255, -1)
                kernels.append((kernel, size_labels[size_idx], ecc_labels[ecc_idx], angle))

                # Salvar o kernel
                kernel_name = f"kernel_scale{size:.2f}_ecc{ecc:.2f}_angle{angle}.npy"
                np.save(os.path.join(output_folder, kernel_name), kernel)
                kernels.append(kernel)

    return output_folder

def visualize_kernels(kernels, grid_size, kernel_size=(200, 200)):
    """
    Cria uma única imagem consolidada para visualizar todos os kernels gerados.

    Args:
    - kernels: lista de kernels.
    - grid_size: dimensão da grade (linhas, colunas).
    - kernel_size: tamanho de cada kernel (altura, largura).

    Retorna:
    - consolidated_image: imagem única consolidada.
    """
    rows, cols = grid_size
    h, w = kernel_size
    consolidated_image = np.zeros((rows * h, cols * w), dtype=np.float32)

    for idx, kernel in enumerate(kernels):
        row = idx // cols
        col = idx % cols
        if row >= rows:
            break
        start_y, start_x = row * h, col * w
        consolidated_image[start_y:start_y + h, start_x:start_x + w] = kernel

    return consolidated_image

# Parâmetros
sizes = [50, 80, 110]  # Tamanhos pequeno, normal e grande
eccentricities = [0.3, 0.6, 0.9]  # Excentricidades baixa, média e alta
angles = list(range(10, 181, 10))  # Ângulos de 0 a 180 graus

# Gerar kernels
output_folder = generate_ellipse_kernels(sizes, eccentricities, angles, output_folder="kernels")

# Carregar kernels gerados
kernel_files = sorted(os.listdir(output_folder))
kernels = [np.load(os.path.join(output_folder, f)) for f in kernel_files]

# Organizar grade para visualização
num_kernels = len(kernels)
grid_cols = 12  # Aumentar número de colunas
grid_rows = math.ceil(num_kernels / grid_cols)

# Criar e salvar a imagem consolidada
consolidated_image = visualize_kernels(kernels, (grid_rows, grid_cols))
plt.imsave("consolidated_kernels_updated.png", consolidated_image, cmap="gray")