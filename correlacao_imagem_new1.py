import cv2
import numpy as np
from scipy.ndimage import maximum_filter, label
import os

# Função para realizar correlação entre a imagem e os kernels
def correlate_with_kernels(image, kernels):
    """
    Realiza a correlação cruzada entre a imagem e um conjunto de kernels.

    Args:
    - image: imagem de entrada (em tons de cinza ou BGR).
    - kernels: lista de kernels elípticos gerados.

    Retorna:
    - correlation_map: mapa de correlação.
    """
    correlation_map = np.zeros_like(image, dtype=np.float32)

    for kernel in kernels:
        # Correlação cruzada usando a função de convolução
        response = cv2.filter2D(image, -1, kernel)
        # Atualiza o mapa de correlação com o valor máximo
        correlation_map = np.maximum(correlation_map, response)

    return correlation_map


# Função para extrair picos do mapa de correlação
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
    
    # Rotulação para agrupar picos conectados
    labeled, num_features = label(peaks)
    
    # Extrair as coordenadas dos picos
    peak_coords = np.array(np.nonzero(peaks)).T  # Coordenadas (y, x)
    
    return peak_coords


# Função para destacar os grãos na imagem original
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
    for i in range(len(peaks)):
        y = peaks[i][0]
        x = peaks[i][1]

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


# Fluxo completo de execução
def main(image_path, kernels):
    """
    Função principal para processar a imagem e destacar os grãos.

    Args:
    - image_path: caminho para a imagem de entrada.
    - kernels: lista de kernels gerados.
    """
    # Carregar a imagem original (em BGR)
    original_image = cv2.imread("C:/Users/jsantos1/OneDrive - QuidelOrtho/Documents/Mestrado/Code/bean_output_new.png")

    # Converter a imagem para escala de cinza para correlação
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # Realizar correlação entre a imagem e os kernels
    correlation_map = correlate_with_kernels(gray_image, kernels)

    # Extrair picos do mapa de correlação
    peaks = extract_peaks(correlation_map, threshold=0.5, min_distance=10)

    # Destacar os grãos na imagem original
    highlighted_image = highlight_grains_on_image(original_image, peaks)

    # Salvar a imagem final com os grãos destacados
    cv2.imwrite("highlighted_grains_final.png", highlighted_image)
    print("Imagem final salva como 'highlighted_grains_final.png'.")


# # Exemplo de uso: Defina os kernels e o caminho da imagem
kernel_files = sorted(os.listdir("kernels"))
kernels = [np.load(os.path.join("kernels", f)) for f in kernel_files]
image_path = "C:/Users/jsantos1/OneDrive - QuidelOrtho/Documents/Mestrado/Code/feijao1.png"  # Substitua pelo caminho correto da sua imagem

# # Rodar o processo
main(image_path, kernels)
