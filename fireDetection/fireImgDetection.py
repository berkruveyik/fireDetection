from ultralytics import YOLO
import cvzone
import cv2
import math


model = YOLO('best.pt')
model.predict(source='./test_ımg/fire2Img.jpg', imgsz = 640, conf = 0.6,save = True)