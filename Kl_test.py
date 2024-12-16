import numpy as np
import cv2
import matplotlib.pyplot as plt

def generate_ellipse_kernels(sizes, eccentricities, angles, image_size=(200, 200)):
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
    return kernels

# Parâmetros
sizes = [50, 80, 110]  # Tamanhos pequeno, normal e grande
eccentricities = [0.3, 0.6, 0.9]  # Excentricidades baixa, média e alta
angles = list(range(0, 181, 10))  # Ângulos de 0 a 180 graus

# Gerar kernels
kernels = generate_ellipse_kernels(sizes, eccentricities, angles)

# Visualizar alguns exemplos
fig, axes = plt.subplots(9, 9, figsize=(5, 2))
axes = axes.ravel()

for i in range(28):  # Mostrar 18 kernels como exemplo
    kernel, size_label, ecc_label, angle = kernels[i]
    axes[i].imshow(kernel, cmap='gray')
    axes[i].set_title(f"{size_label}, {ecc_label}, {angle}°")
    axes[i].axis('off')

plt.tight_layout()
plt.show()

len(kernels)  # Total de kernels gerados
