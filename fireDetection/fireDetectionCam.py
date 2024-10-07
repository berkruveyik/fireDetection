from ultralytics import YOLO
import cvzone
import cv2
import math

# Webcam'den video akışını başlatma
cap = cv2.VideoCapture(0)  # 0, varsayılan kamerayı temsil eder; harici bir kamera kullanıyorsanız 1 veya 2 olarak değiştirin

# Modeli yükleme
model = YOLO('best.pt')

# Sınıf isimleri (modelinize uygun olarak güncelleyin)
classnames = ['fire', 'other', 'smoke']

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Frame boyutunu ayarlama
    frame = cv2.resize(frame, (1280, 768))
    
    # Model ile tahmin yapma
    results = model(frame, stream=True)

    # Tespit edilen nesneleri işleme
    for result in results:
        boxes = result.boxes
        for box in boxes:
            confidence = box.conf[0]
            confidence = math.ceil(confidence * 100)  # Güven skoru yüzdesi

            class_id = int(box.cls[0])
            if confidence > 45:  # Güven eşiği
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Koordinatları al
                
                # Sınıfa göre rengi değiştir
                if class_id == 0:  # fire
                    color = (0, 0, 255)  # Kırmızı
                elif class_id == 1:  # other
                    color = (0, 255, 0)  # Yeşil
                elif class_id == 2:  # smoke
                    color = (255, 0, 0)  # Mavi

                # Çerçeve çiz ve metni yerleştir
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 5)
                cvzone.putTextRect(frame, f'{classnames[class_id]} {confidence}%', [x1 + 8, y1 - 10],
                                   scale=1.5, thickness=2, colorR=color)

    # Görüntüyü göster
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Video kaynağını serbest bırak ve pencereleri kapat
cap.release()
cv2.destroyAllWindows()
