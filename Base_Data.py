import cv2
import os
import imutils

personName = 'inge_Roldan'
dataPath = 'B:/ReconocimientoFacial/Data'
personPath = os.path.join(dataPath, personName)

if not os.path.exists(personPath):
    print('Carpeta Creada:', personPath)
    os.makedirs(personPath)

# Cambia esta línea:
# cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Cámara integrada (generalmente índice 0)
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # Cámara USB (generalmente índice 1)
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
count = 0
target_size = (200, 200)  # Mismo tamaño que en entrenamiento

if not cap.isOpened():
    print("Error: No se pudo abrir la cámara")
    exit()

try:
    while count < 150:  # Capturar 300 imágenes
        ret, frame = cap.read()
        if not ret: break
        
        frame = imutils.resize(frame, width=320)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        auxFrame = gray.copy()

        faces = faceClassif.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            rostro = auxFrame[y:y+h, x:x+w]
            rostro = cv2.resize(rostro, target_size)  # Usar tamaño consistente
            cv2.imwrite(os.path.join(personPath, f'rostro_{count}.jpg'), rostro)
            count += 1

        cv2.imshow('Capturando Rostros', frame)
        if cv2.waitKey(1) == 27 or cv2.getWindowProperty('Capturando Rostros', cv2.WND_PROP_VISIBLE) < 1:
            break

finally:
    cap.release()
    cv2.destroyAllWindows()