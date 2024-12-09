import numpy as np
import matplotlib.pyplot as plt

def generate_elliptical_kernel(size, angle):
    """
    Gera um kernel elíptico com uma orientação específica.
    
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
    
    # Equação de uma elipse
    ellipse = (x_rot**2 / (w//3)**2) + (y_rot**2 / (h//4)**2) <= 1
    return ellipse.astype(float)

def generate_kernel_grid(rows, cols, kernel_size):
    """
    Gera uma grade de kernels elípticos com diferentes ângulos.
    
    Args:
    - rows: número de linhas na grade.
    - cols: número de colunas na grade.
    - kernel_size: tamanho de cada kernel (altura, largura).
    
    Retorna:
    - grid_image: imagem 2D da grade de kernels.
    """
    angle_step = 180 // cols  # Passo de ângulo para cada coluna
    grid_height = rows * kernel_size[0]
    grid_width = cols * kernel_size[1]
    
    grid_image = np.zeros((grid_height, grid_width))
    
    for i in range(rows):
        for j in range(cols):
            angle = j * angle_step  # Incrementar o ângulo com base na coluna
            kernel = generate_elliptical_kernel(kernel_size, angle)
            
            # Inserir o kernel na posição correspondente da grade
            start_y = i * kernel_size[0]
            start_x = j * kernel_size[1]
            grid_image[start_y:start_y + kernel_size[0], start_x:start_x + kernel_size[1]] = kernel
    
    return grid_image

# Configurações da grade
rows, cols = 6, 20  # Número de linhas e colunas
kernel_size = (40, 40)  # Tamanho de cada kernel

# Gerar a grade de kernels
grid_image = generate_kernel_grid(rows, cols, kernel_size)

# Mostrar e salvar a imagem
plt.figure(figsize=(15, 5))
plt.imshow(grid_image, cmap="gray")
plt.axis("off")
plt.savefig("kernel_grid.png", dpi=300, bbox_inches="tight")
plt.show()
