# Importing libraries

import cv2
from ultralytics import YOLO
import numpy as np
import torch


print("MPS is available: ", torch.backends.mps.is_available())


cap = cv2.VideoCapture("MartialArts.mp4")
model = YOLO("yolov8m.pt")
while True:
    ret, frame = cap.read()
    #print('Completed reading video file')
    if not ret:
        break
    results = model(frame, device="mps")
    #print(results)
    result = results[0]
    bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
    classes = np.array(result.boxes.cls.cpu(), dtype="int")
    #print(bboxes)
    for bbox,cls in zip(bboxes,classes):
        (x, y, x2, y2) = bbox
        cv2.rectangle(frame, (x,y), (x2,y2), (0,0,225), 2)
        cv2.putText(frame, result.names[cls], (x,y - 5), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,225), 2)
    cv2.imshow("Img", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()


"""
CPU to GPU boost was observed to be 5x, from 4 fps to 20 fps.
"""