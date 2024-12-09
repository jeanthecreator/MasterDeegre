import numpy as np
import cv2
import os
from scipy.ndimage import maximum_filter, label

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
    correlation_maps = np.zeros_like(image, dtype=np.float32)

    for kernel in kernels:
        # Correlação cruzada com cada kernel
        response = cv2.filter2D(image, -1, kernel, anchor=(-1, -1), delta=0, borderType=cv2.BORDER_WRAP)
        correlation_maps = np.maximum(correlation_maps, response)


    return correlation_maps

def extract_peaks(correlation_map, threshold=0.5, min_distance=10):
    """
    Extrai picos locais do mapa de correlação, que representam os grãos.

    Args:
    - correlation_map: mapa de correlação (2D).
    - threshold: valor mínimo de correlação para considerar um pico.
    - min_distance: distância mínima entre picos consecutivos.

    Retorna:
    - peaks: lista de coordenadas (y, x) dos picos.
    """
    # Normalizar o mapa de correlação para intervalo [0, 1]
    correlation_map_normalized = correlation_map / correlation_map.max()

    # Localizar picos locais: onde o valor é maior que os vizinhos
    peaks = maximum_filter(correlation_map_normalized, size=min_distance) == correlation_map_normalized
    
    # Aplicar o limiar de intensidade: apenas picos com alta correlação
    peaks = peaks & (correlation_map_normalized > threshold)
    
    # Extrair as coordenadas dos picos
    peak_coords = np.column_stack(np.nonzero(peaks))  # Coordenadas (y, x)
    
    return peak_coords

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
    for y, x in peaks:
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

# Caminho para a imagem de entrada
input_image_path = "C:/Users/jsantos1/OneDrive - QuidelOrtho/Documents/Mestrado/Code/feijao1.png"
image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)

# Carregar os kernels gerados
kernel_files = sorted(os.listdir("kernels"))
kernels = [np.load(os.path.join("kernels", f)) for f in kernel_files]

# Aplicar a correlação
correlation_map = correlate_with_kernels(image, kernels)

# Normalizar e salvar o mapa de correlação
correlation_map_normalized = cv2.normalize(correlation_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
cv2.imwrite("correlation_map.png", correlation_map_normalized)

# Extrair os picos do mapa de correlação
peaks = extract_peaks(correlation_map, threshold=0.5, min_distance=10)

# Carregar a imagem original (em cores)
original_image = cv2.imread(input_image_path)

# Destacar os grãos na imagem original
highlighted_image = highlight_grains_on_image(original_image, peaks)

# Salvar a imagem final com os grãos destacados
cv2.imwrite("highlighted_grains_final.png", highlighted_image)
