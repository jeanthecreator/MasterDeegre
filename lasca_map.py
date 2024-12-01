from pathlib import Path
import cv2
import numpy as np
from Image_input import import_image
from Image_input import extract_r_channel
from image_processing import image_mask, apply_mask

# def calc_lasca(image):

#     if image is None:
#         print("Image not found. ")
#         return None
    
#     media = np.mean(image)
#     desv_pd = np.std(image)

#     contrast = desv_pd / media

#     return contrast

caminho = Path(r"C:/Users/jsantos1/OneDrive - QuidelOrtho/Documents/Mestrado/Documentos gerais/Sidnei/IMG_8352.JPG")
imagem = import_image(caminho)
#imagem_tc = cv2.resize(imagem, (500, 500))

# mapa_vermelho = extract_r_channel(imagem_tc)
definir_mask = image_mask(imagem)
aplicar_mask = apply_mask(definir_mask)

if imagem is not None:
     cv2.imshow(aplicar_mask)

# if imagem is not None:
#     cv2.imshow("Imagem", imagem)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# if mapa_vermelho is not None:
#     cv2.imshow("Mapa do Canal Vermelho", mapa_vermelho)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
