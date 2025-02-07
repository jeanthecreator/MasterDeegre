import cv2
import numpy as np
import os
import math

def rotate_image(image, angle, scale=1.0):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

def mlp_training(training_data, training_classes):
    mlp = cv2.ml.ANN_MLP_create()
    layer_sizes = np.array([20, 45, 3], dtype=np.int32)
    mlp.setLayerSizes(layer_sizes)
    mlp.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM, 0.0, 0.0)
    mlp.setTrainMethod(cv2.ml.ANN_MLP_BACKPROP, 0.05)
    
    criteria = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 2000, 0.00001)
    mlp.setTermCriteria(criteria)
    
    train_data = cv2.ml.TrainData_create(training_data, cv2.ml.ROW_SAMPLE, training_classes)
    mlp.train(train_data)
    
    mlp.save("mlp.yml")
    print("Treinamento MLP concluído e salvo.")

def mlp_test(test_data):
    mlp = cv2.ml.ANN_MLP_load("mlp.yml")
    _, response = mlp.predict(test_data)
    
    prediction = np.argmax(response, axis=1)
    
    labels = {0: "Carioca", 1: "Mulato", 2: "Preto"}
    return labels.get(prediction[0], "Indefinido")

def process_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Erro ao abrir imagem {image_path}")
        return None
    
    img = img.astype(np.float32) / 255.0
    img_cie = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    
    features = np.zeros((1, 20), dtype=np.float32)
    qtd_circ = 4
    passo_angulo = 2
    passo_raio = 3
    raio_inicial = 3
    
    for qc in range(qtd_circ):
        r = raio_inicial + qc * passo_raio
        for theta in range(0, 360, passo_angulo):
            x = int(r * np.cos(np.radians(theta)) + img.shape[1] / 2)
            y = int(r * np.sin(np.radians(theta)) + img.shape[0] / 2)
            
            features[0, qc * 3] += img_cie[y, x, 0]
            features[0, qc * 3 + 1] += img_cie[y, x, 1] + 128
            features[0, qc * 3 + 2] += img_cie[y, x, 2] + 128
    
    return features

def apply_ncc(image, template):
    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    return max_loc, max_val

def segment_grains(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 128, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def classify_grains(image):
    contours = segment_grains(image)
    classification = {"carioca": 0, "mulato": 0, "preto": 0}
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:
            classification["carioca"] += 1
        elif area > 300:
            classification["mulato"] += 1
        else:
            classification["preto"] += 1
    
    return classification

if __name__ == "__main__":
    training_data = np.load("data_train.npy")
    training_classes = np.load("data_train_class.npy")
    mlp_training(training_data, training_classes)
    
    test_image = "test_image.png"
    img = cv2.imread(test_image)
    if img is not None:
        classification = classify_grains(img)
        print("Classificação dos grãos:", classification)
