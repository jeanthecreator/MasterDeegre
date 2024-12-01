from pathlib import Path
import cv2
import numpy as np

def import_image(image_path):
    
    #converter 0 caminho para path
    image_path = Path(image_path)

    #Verificar o arquivo
    if not image_path.is_file():
        print("File not found. ")
        return None

    #Ler a imagem com OpenCV
    image = cv2.imread(str(image_path))

    if image is not None:
        print("Image imported Successfuly. ")
    else:
        print("Error on Import image")
    
    return image

def extract_r_channel(image):

    if image is None:
        print("Image not found. ")
        return None

    #Extrair R
    red_channel = image[:, :, 2]

    return red_channel

def import_image_directory(directory, extention=['.jpg', '.jpeg', '.png', '.bmp', '.tiff'] ):

    #converter 0 caminho para path
    directory = Path(directory)

    #Verificar se o diretório existe
    if not directory.is_dir():
        print("Folder not found. ")
        return[]
    
    #Lista para armazenar imagens
    images = []

    #Procurar pelas imagens no diretorio

    for ext in extention:
        for image_path in directory.glob('*' + ext):
            
            #carregar imagem com OpenCV
            image = cv2.imread(str(image_path))
            if image is not None:
                images.append(image)
                print(f"Image{image_path.name} loaded successfuly")
            else:
                print(f"Erro on load image{image_path.name}")
    
    return images