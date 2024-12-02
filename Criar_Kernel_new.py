import cv2
import numpy as np
import os
import math
import matplotlib.pyplot as plt

def create_kernels(scales, eccentricities, angles, size=(31, 31), output_folder="kernels"):
    """
    Gera e salva kernels elípticos para segmentação baseada em correlação.

    Args:
    - scales: lista de tamanhos relativos (0 a 1).
    - eccentricities: lista de razões de excentricidade (0 a 1).
    - angles: lista de ângulos de rotação (em graus).
    - size: dimensão do kernel (deve ser ímpar, como 31x31).
    - output_folder: pasta onde os kernels serão salvos.

    Retorna:
    - kernels: lista de kernels gerados.
    """
    kernels = []
    h, w = size
    center = (w // 2, h // 2)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for scale in scales:
        for ecc in eccentricities:
            for angle in angles:
                # Criar uma máscara elíptica
                mask = np.zeros((h, w), dtype=np.float32)
                axes = (int(scale * w // 2), int(scale * w // 2 * ecc))
                cv2.ellipse(mask, center, axes, angle, 0, 360, 1, -1)

                # Balancear áreas positivas e negativas
                positive = mask > 0
                mask[positive] = 0.5 / np.sum(positive)  # Pixels brancos: +0.5
                mask[~positive] = -0.5 / np.sum(~positive)  # Pixels pretos: -0.5

                # Salvar o kernel
                kernel_name = f"kernel_scale{scale:.2f}_ecc{ecc:.2f}_angle{angle}.npy"
                np.save(os.path.join(output_folder, kernel_name), mask)
                kernels.append(mask)

    return output_folder

def visualize_kernels(kernels, grid_size, kernel_size=(31, 31)):
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

# Parâmetros ajustados para mais combinações de kernels
scales = np.linspace(0.5, 1.0, 5)  # Mais passos para escala
eccentricities = np.linspace(0.5, 1.0, 1)  # Mais passos para excentricidade
angles = np.arange(90, 270, 10)  # Ângulos com intervalos menores

# Geração dos kernels
output_folder = create_kernels(scales, eccentricities, angles, output_folder="kernels")

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
