import numpy as np
import os
import cv2
import math
import matplotlib.pyplot as plt

def generate_elliptical_kernel_with_borders(size, angle):
    """
    Gera um kernel elíptico com borda branca, centro preto e fundo cinza.

    Args:
    - size: tamanho do kernel (altura, largura).
    - angle: ângulo de rotação do elipse em graus.

    Retorna:
    - kernel: imagem 2D representando o kernel elíptico.
    """
    h, w = size
    y, x = np.ogrid[-h//2:h//2, -w//2:w//2]
    
    # Rotação do elipse
    theta = np.deg2rad(angle)
    x_rot = x * np.cos(theta) - y * np.sin(theta)
    y_rot = x * np.sin(theta) + y * np.cos(theta)
    
    # Criar o centro preto da elipse
    inner_ellipse = (x_rot**2 / (w//7)**2) + (y_rot**2 / (h//3)**2) <= 1
    
    # Criar a borda branca ao redor
    outer_ellipse = (x_rot**2 / (w//6)**2) + (y_rot**2 / (h//2.5)**2) <= 1
    
    # Construir o kernel com três camadas: cinza (fundo), preto (centro) e branco (borda)
    kernel = np.zeros_like(x_rot, dtype=float) + 0.5  # Fundo cinza
    kernel[outer_ellipse] = 0.0  # Borda branca
    kernel[inner_ellipse] = 1.0  # Centro preto

    return kernel

def create_kernels(scales, eccentricities, angles, size, output_folder="kernels"):
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
                # Gerar o kernel com bordas usando a função anterior
                kernel = generate_elliptical_kernel_with_borders(size, angle)

                # Ajustar escala e excentricidade
                mask = np.zeros((h, w), dtype=np.float32)
                axes = (int(scale * w // 2), int(scale * w // 2 * ecc))
                cv2.ellipse(mask, center, axes, angle, 0, 360, 1, -1)

                # Balancear áreas positivas e negativas
                positive = mask > 0
                mask[positive] = 0.5 / np.sum(positive)  # Pixels brancos: +0.5
                mask[~positive] = -0.5 / np.sum(~positive)  # Pixels pretos: -0.5

                # Salvar o kernel
                kernel_name = f"kernel_scale{scale:.2f}_ecc{ecc:.2f}_angle{angle}.npy"
                np.save(os.path.join(output_folder, kernel_name), kernel)
                kernels.append(kernel)

    return output_folder

def visualize_kernels(kernels, grid_size, kernel_size=(41, 41)):
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

# Parâmetros de geração
scales = np.linspace(0.8, 1.6, 6)#[0.6, 0.8, 1.0]  # Escalas dos kernels
eccentricities = np.linspace(0.5, 1.0, 1)#[0.5, 0.7, 1.0]  # Excentricidades
angles = np.arange(90, 280, 10)#[0, 45, 90]  # Ângulos de rotação
size = (41, 41)  # Tamanho do kernel (deve ser ímpar)

# Gerar e salvar kernels
output_folder = create_kernels(scales, eccentricities, angles, size)

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