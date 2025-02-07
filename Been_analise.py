import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# Rotacionar imagem
def rotate_image(image, angle, scale=1.0):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

# Carregar imagem
def load_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Erro: O arquivo '{image_path}' não foi encontrado.")
    
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Erro: Não foi possível carregar a imagem '{image_path}'. Verifique o caminho e a integridade do arquivo.")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

# Criar máscara do fundo
def create_background_mask(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lower_white = np.array([0, 0, 180], dtype=np.uint8)
    upper_white = np.array([180, 30, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv_image, lower_white, upper_white)
    binary_image = np.ones_like(mask) * 255
    binary_image[mask > 0] = 255  # Fundo branco
    binary_image[mask == 0] = 0  # Grãos pretos
    return binary_image

# Carregar kernels
def load_samples_from_folder(folder_path, kernel_size=(24, 24)):
    samples = []
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Erro: A pasta '{folder_path}' não foi encontrada.")
    
    for filename in os.listdir(folder_path):
        sample_image = cv2.imread(os.path.join(folder_path, filename), cv2.IMREAD_GRAYSCALE)
        if sample_image is not None:
            resized_kernel = cv2.resize(sample_image, kernel_size)
            samples.append(resized_kernel)
    
    if not samples:
        raise ValueError("Erro: Nenhum kernel foi carregado corretamente. Verifique a pasta de kernels.")
    
    return samples

# Aplicação de convolução
def apply_convolution(binary_image, kernels):
    response = np.zeros_like(binary_image, dtype=np.float32)
    
    for kernel in kernels:
        conv_result = cv2.filter2D(binary_image, -1, kernel)
        response = np.maximum(response, conv_result)  # Mantém a maior resposta da convolução
    
    response = cv2.normalize(response, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, response = cv2.threshold(response, 50, 255, cv2.THRESH_BINARY)
    return response

# Segmentação de grãos
def segment_grains(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 128, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

# Criar bordas
def extract_edges(response):
    edges = cv2.Canny(response, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    return edges

# Sobreposição de bordas
def overlay_edges_on_original(image, edge_image):
    result_image = image.copy()
    edge_coords = np.where(edge_image == 255)
    result_image[edge_coords[0], edge_coords[1]] = [0, 0, 255]  # Azul
    return result_image

# Pipeline
image_path = "C:/Users/jsantos1/OneDrive - QuidelOrtho/Documents/Mestrado/Code/Images/feijao1.png" # Substitua pela imagem real
kernel_folder = "C:/Users/jsantos1/OneDrive - QuidelOrtho/Documents/Mestrado/Code/kernel_paths/"  # Pasta com os kernels

try:
    image = load_image(image_path)
    binary_image = create_background_mask(image)
    kernels = load_samples_from_folder(kernel_folder, kernel_size=(24, 24))
    response = apply_convolution(binary_image, kernels)
    edge_image = extract_edges(response)
    final_result = overlay_edges_on_original(image, edge_image)
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 4, 1)
    plt.imshow(image)
    plt.title("Imagem Original")
    plt.subplot(1, 4, 2)
    plt.imshow(binary_image, cmap='gray')
    plt.title("Máscara de Fundo")
    plt.subplot(1, 4, 3)
    plt.imshow(edge_image, cmap='gray')
    plt.title("Imagem com Bordas")
    plt.subplot(1, 4, 4)
    plt.imshow(final_result)
    plt.title("Sobreposição Final")
    plt.show()

except Exception as e:
    print(e)
