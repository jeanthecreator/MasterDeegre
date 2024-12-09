import numpy as np
import os

# Diretório onde os arquivos .npy estão armazenados
kernels_folder = "C:/Users/jsantos1/OneDrive - QuidelOrtho/Documents/Mestrado/Code/kernels"

# Carregar todos os arquivos .npy na pasta
kernel_files = [f for f in os.listdir(kernels_folder) if f.endswith('.npy')]

kernels = []
for file in kernel_files:
    kernel_path = os.path.join(kernels_folder, file)
    kernel = np.load(kernel_path)  # Carregar o arquivo .npy
    kernels.append(kernel)

# Agora você tem todos os kernels carregados na lista `kernels`
print(kernels)  # Exibe os kernels carregados