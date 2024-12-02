import cv2
import numpy as np
from scipy.ndimage import maximum_filter, label
import os

def correlate_with_kernels(image, kernels):
    """
    Realiza correlação cruzada entre a imagem e um conjunto de kernels, retornando o mapa máximo de correlação.

    Args:
    - image: imagem binarizada ou em tons de cinza.
    - kernels: lista de kernels.

    Retorna:
    - correlation_map: mapa de correlação máxima.
    """
    # Lista para armazenar os mapas de correlação
    correlation_maps = []

    for kernel in kernels:
        # Correlação cruzada com cada kernel
        response = cv2.filter2D(image, -1, kernel)
        correlation_maps.append(response)

    # Seleção do máximo
    correlation_map = np.max(correlation_maps, axis=0)

    return correlation_map

# Carregar a imagem de entrada (em tons de cinza)
input_image_path = "C:/Users/jsantos1/OneDrive - QuidelOrtho/Documents/Mestrado/Code/feijao1.png"  # Substitua pelo caminho da sua imagem
image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)

# Carregar os kernels gerados
kernel_files = sorted(os.listdir("kernels"))
kernels = [np.load(os.path.join("kernels", f)) for f in kernel_files]

# Aplicar a correlação
correlation_map = correlate_with_kernels(image, kernels)

# Normalizar e salvar o mapa de correlação
correlation_map_normalized = cv2.normalize(correlation_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
cv2.imwrite("correlation_map.png", correlation_map_normalized)

def extract_peaks(correlation_map, threshold=0.5, min_distance=10):
    """
    Extrai picos locais do mapa de correlação, que representam os grãos.

    Args:
    - correlation_map: mapa de correlação (2D).
    - threshold: valor mínimo de correlação para considerar um pico.
    - min_distance: distância mínima entre picos consecutivos.

    Retorna:
    - peaks: lista de coordenadas (x, y) dos picos.
    """
    # Localizar picos locais: onde o valor é maior que os vizinhos
    peaks = maximum_filter(correlation_map, size=min_distance) == correlation_map
    
    # Aplicar o limiar de intensidade: apenas picos com alta correlação
    peaks = peaks & (correlation_map > threshold)
    
    # Label para agrupar picos conectados
    labeled, num_features = label(peaks)
    
    # Extrair as coordenadas dos picos
    peak_coords = np.array(np.nonzero(peaks)).T  # Coordenadas (y, x)
    
    return peak_coords

# Carregar a imagem original
original_image = cv2.imread("C:/Users/jsantos1/OneDrive - QuidelOrtho/Documents/Mestrado/Code/feijao1.png")

# Aplicar correlação com os kernels
correlation_map = correlate_with_kernels(original_image, kernels)

# Extrair os picos do mapa de correlação
peaks = extract_peaks(correlation_map, threshold=0.5, min_distance=10)

def highlight_grains_on_image(image, peaks, kernel_size=(31, 31)):
    """
    Destaca os grãos diretamente na imagem original com base nos picos detectados.

    Args:
    - image: imagem original (em BGR ou RGB).
    - peaks: coordenadas dos picos extraídos do mapa de correlação.
    - kernel_size: tamanho dos kernels aplicados (altura, largura).

    Retorna:
    - highlighted_image: imagem com os grãos destacados.
    """
    h, w = kernel_size
    half_h, half_w = h // 2, w // 2

    # Copiar a imagem original
    highlighted_image = image.copy()

    # Iterar pelas coordenadas (y, x)
    for i in range(len(peaks[0])):
        y = peaks[0][i]
        x = peaks[1][i]

        # Definir a região ao redor do pico
        y_start = max(0, y - half_h)
        y_end = min(image.shape[0], y + half_h)
        x_start = max(0, x - half_w)
        x_end = min(image.shape[1], x + half_w)

        # Aplicar um contorno na área identificada
        cv2.rectangle(
            highlighted_image, 
            (x_start, y_start), 
            (x_end, y_end), 
            (0, 255, 0), 2  # Cor verde e espessura do contorno
        )

    return highlighted_image


# Destacar os grãos na imagem original
highlighted_image = highlight_grains_on_image(original_image, peaks)

# Salvar a imagem final com os grãos destacados
cv2.imwrite("highlighted_grains_final.png", highlighted_image)

# def extract_peaks(correlation_map, threshold=0.2, min_distance=10):
#     """
#     Extrai os picos do mapa de correlação como candidatos a grãos.

#     Args:
#     - correlation_map: mapa de correlação (float32).
#     - threshold: valor mínimo de correlação para considerar um pico.
#     - min_distance: distância mínima entre os picos.

#     Retorna:
#     - peaks: lista de coordenadas (x, y) dos picos.
#     """
#     # Localizar picos locais
#     peaks = maximum_filter(correlation_map, size=min_distance) == correlation_map
#     peaks = peaks & (correlation_map > threshold)
#     labeled, _ = label(peaks)
#     peak_coords = np.array(np.nonzero(peaks)).T  # Coordenadas (y, x)

#     return peak_coords

# # Extrair os picos do mapa de correlação
# threshold = 0.2  # Valor mínimo de intensidade para um pico (ajustável)
# min_distance = 10  # Distância mínima entre picos
# peaks = extract_peaks(correlation_map, threshold, min_distance)

# # Visualizar os picos na imagem original
# output_with_peaks = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
# for y, x in peaks:
#     cv2.circle(output_with_peaks, (x, y), 5, (0, 0, 255), -1)  # Desenhar picos como círculos

# # Salvar a imagem com os picos
# cv2.imwrite("output_with_peaks.png", output_with_peaks)




# # 1. Carregar a imagem original
# original_image = cv2.imread("C:/Users/jsantos1/OneDrive - QuidelOrtho/Documents/Mestrado/Code/feijao1.png")

# # 2. Extrair picos (supondo que o mapa de correlação já foi gerado)
# peaks = extract_peaks(correlation_map, threshold=0.2, min_distance=10)

# # 3. Destacar os grãos na imagem
# highlighted_image = highlight_grains_on_image(original_image, peaks)

# # 4. Salvar a saída final
# cv2.imwrite("highlighted_grains_final.png", highlighted_image)


# def highlight_grains(image, correlation_map, peaks, kernel_size=(31, 31), threshold=0.5):
#     """
#     Destaca os grãos detectados na imagem original aplicando máscaras diretamente.

#     Args:
#     - image: imagem original (em BGR ou RGB).
#     - correlation_map: mapa de correlação gerado.
#     - peaks: coordenadas dos picos extraídos do mapa de correlação.
#     - kernel_size: tamanho dos kernels aplicados.
#     - threshold: limite de intensidade para considerar uma correspondência.

#     Retorna:
#     - highlighted_image: imagem com grãos destacados.
#     """
#     h, w = kernel_size
#     half_h, half_w = h // 2, w // 2
#     highlighted_image = np.zeros_like(image) + 128  # Fundo cinza

#     for y, x in peaks:
#         # Definir a área ao redor do pico
#         y_start = max(0, y - half_h)
#         y_end = min(image.shape[0], y + half_h)
#         x_start = max(0, x - half_w)
#         x_end = min(image.shape[1], x + half_w)

#         # Verificar intensidade no mapa de correlação
#         region = correlation_map[y_start:y_end, x_start:x_end]
#         if np.max(region) < threshold:
#             continue  # Ignorar áreas com baixa intensidade

#         # Copiar pixels da imagem original para os grãos detectados
#         highlighted_image[y_start:y_end, x_start:x_end] = image[y_start:y_end, x_start:x_end]

#     return highlighted_image


# # Aplicar a função revisada
# peaks = extract_peaks(correlation_map, threshold=0.2, min_distance=10)
# highlighted_image = highlight_grains(cv2.imread("C:/Users/jsantos1/OneDrive - QuidelOrtho/Documents/Mestrado/Code/feijao1.png"), correlation_map, peaks)

# # Salvar o resultado final
# cv2.imwrite("highlighted_grains.png", highlighted_image)


# def apply_kernel_mask(image, correlation_map, peaks, kernel_size=(31, 31)):
#     """
#     Aplica máscaras dos kernels diretamente na imagem original, destacando os grãos.

#     Args:
#     - image: imagem original (em BGR ou RGB).
#     - correlation_map: mapa de correlação gerado.
#     - peaks: coordenadas dos picos extraídos do mapa de correlação.
#     - kernel_size: tamanho dos kernels aplicados.

#     Retorna:
#     - image_with_masks: imagem com máscaras aplicadas.
#     """
#     h, w = kernel_size
#     half_h, half_w = h // 2, w // 2
#     image_with_masks = image.copy()

#     for y, x in peaks:
#         # Definir a área ao redor do pico
#         y_start = max(0, y - half_h)
#         y_end = min(image.shape[0], y + half_h)
#         x_start = max(0, x - half_w)
#         x_end = min(image.shape[1], x + half_w)

#         # Aplicar a máscara na região correspondente
#         region = correlation_map[y_start:y_end, x_start:x_end]
#         mask = (region > 0.5).astype(np.uint8)  # Cria uma máscara binária simples
#         mask = cv2.resize(mask, (x_end - x_start, y_end - y_start))  # Ajusta a máscara ao tamanho da região

#         # Destacar os grãos mantendo as cores
#         image_with_masks[y_start:y_end, x_start:x_end] = cv2.addWeighted(
#             image_with_masks[y_start:y_end, x_start:x_end], 0.5,
#             np.dstack([mask * 255] * 3), 0.5, 0
#         )

#     return image_with_masks


# # Aplicar o processo
# peaks = extract_peaks(correlation_map, threshold=0.2, min_distance=10)
# image_with_masks = apply_kernel_mask(cv2.imread("C:/Users/jsantos1/OneDrive - QuidelOrtho/Documents/Mestrado/Code/feijao1.png"), correlation_map, peaks)

# # Salvar o resultado final
# cv2.imwrite("image_with_masks.png", image_with_masks)