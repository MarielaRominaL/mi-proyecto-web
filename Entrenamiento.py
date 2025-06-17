import cv2
import os
import numpy as np

dataPath = 'B:/ReconocimientoFacial/Data'
peopleList = os.listdir(dataPath)
print('Personas registradas:', peopleList)

labels = []
facesData = []
label = 0
target_size = (200, 200)  # Mismo tamaño que en captura

for nameDir in peopleList:
    personPath = os.path.join(dataPath, nameDir)
    print('Procesando:', nameDir)

    for fileName in os.listdir(personPath):
        filePath = os.path.join(personPath, fileName)
        image = cv2.imread(filePath, 0)
        
        if image is None:
            continue
            
        image = cv2.resize(image, target_size)
        image = cv2.equalizeHist(image)
        facesData.append(image)
        labels.append(label)

    label += 1

# Usar FisherFace que es mejor que EigenFace
face_recognizer = cv2.face.FisherFaceRecognizer_create()
face_recognizer.train(facesData, np.array(labels))

modelPath = 'ModeloFaceFrontalData2025.xml'
face_recognizer.write(modelPath)
print(f'Modelo guardado en {modelPath} con {len(facesData)} imágenes')