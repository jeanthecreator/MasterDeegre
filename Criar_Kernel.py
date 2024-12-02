import cv2
import numpy as np
import os

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
                kernel_name = f"kernel_scale{scale}_ecc{ecc}_angle{angle}.npy"
                np.save(os.path.join(output_folder, kernel_name), mask)
                kernels.append(mask)

    print(f"Kernels salvos na pasta: {output_folder}")
    return kernels

# Parâmetros para os kernels
scales = [0.6, 0.8, 1.0]  # Tamanhos relativos
eccentricities = [0.6, 0.8, 1.0]  # Excentricidade
angles = range(0, 180, 20)  # Ângulos de rotação

# Geração dos kernels
kernels = create_kernels(scales, eccentricities, angles)
