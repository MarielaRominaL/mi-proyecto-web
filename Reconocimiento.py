import cv2
import os

dataPath = 'B:/ReconocimientoFacial/Data'
imagePaths = os.listdir(dataPath)
print('Personas registradas:', imagePaths)

face_recognizer = cv2.face.FisherFaceRecognizer_create()
face_recognizer.read('ModeloFaceFrontalData2025.xml')

# Cambia esta línea:
# cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Cámara integrada (generalmente índice 0)
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # Cámara USB (generalmente índice 1)
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
target_size = (200, 200)  # Mismo tamaño que en entrenamiento

while True:
    ret, frame = cap.read()
    if not ret: break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceClassif.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        rostro = gray[y:y+h, x:x+w]
        rostro = cv2.resize(rostro, target_size)
        rostro = cv2.equalizeHist(rostro)
        
        label, confidence = face_recognizer.predict(rostro)
        
        if confidence < 500:  # Umbral ajustado para FisherFace
            cv2.putText(frame, imagePaths[label], (x, y-20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'Desconocido', (x, y-20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('Reconocimiento Facial', frame)
    if cv2.waitKey(1) == 27 or cv2.getWindowProperty('Reconocimiento Facial', cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()