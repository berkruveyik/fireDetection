from ultralytics import YOLO
import cvzone
import cv2
import math

# Video dosyasını açma
cap = cv2.VideoCapture('./test_ımg/fire2.mp4')

# Modeli yükleme
model = YOLO('best.pt')

# Sınıf isimleri (modelinize uygun olarak güncelleyin)
classnames = ['fire', 'smoke', 'others']

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Frame boyutunu ayarlama
    frame = cv2.resize(frame, (640, 480))
    
    # Model ile tahmin yapma
    results = model(frame, stream=True)

    # Tespit edilen nesneleri işleme
    for info in results:
        boxes = info.boxes
        for box in boxes:
            confidence = box.conf[0]
            confidence = math.ceil(confidence * 100)  # Güven skoru yüzdesi

            class_id = int(box.cls[0])
            if confidence > 50:  # Güven eşiği
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Koordinatları al
                color = (0, 0, 255)  # Varsayılan renk (kırmızı)

                # Sınıfa göre rengi değiştir
                if class_id == 1:  # Smoke
                    color = (255, 0, 0)  # Mavi
                elif class_id == 2:  # Others
                    color = (0, 255, 0)  # Yeşil

                # Çerçeve çiz ve metni yerleştir
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 5)
                cvzone.putTextRect(frame, f'{classnames[class_id]} {confidence}%', [x1 + 8, y1 - 10],
                                   scale=1.5, thickness=2)

    # Görüntüyü göster
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Video kaynağını serbest bırak ve pencereleri kapat
cap.release()
cv2.destroyAllWindows()
