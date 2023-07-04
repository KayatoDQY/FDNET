from ultralytics import YOLO
import os
import cv2
import torch
from torch import nn
import numpy as np

from model import FDNET

yolomodel = YOLO("model/yolo/yolov8n-pose.pt")
fdnet = FDNET()
fdnet = fdnet.to(torch.device("cuda"))
m_state_dict = torch.load("model/FDNET.pt")
fdnet.load_state_dict(m_state_dict)

# fourcc = cv2.VideoWriter_fourcc(*'XVID')

# writer = cv2.VideoWriter("result.mp4",fourcc, 30.0, (640,384),True)
sourcename = 0  # 摄像头
# sourcename="video/source.mp4"
cap = cv2.VideoCapture(sourcename)
while cap.isOpened():
    success, frame = cap.read()
    if success:
        results = yolomodel(frame, conf=0.6)
        annotated_frame = results[0].plot()
        # print(results[0].keypoints.xyn.reshape(17 * 2).shape)
        print(results[0].keypoints.xyn.shape[0])
        boxslist = results[0].boxes.xyxy.tolist()
        for pnum in range(results[0].keypoints.xyn.shape[0]):
            keypoint = results[0].keypoints.xyn[pnum]
            if keypoint.shape[0] == 17:
                keypoint = keypoint.reshape(34).unsqueeze(0).cuda()
                output = fdnet.forward(keypoint)
                output = output.tolist()
                if output[0][0] > output[0][1]:
                    mark = "fall down"
                elif output[0][0] < output[0][1]:
                    mark = "stand"
                x_min, y_min, x_max, y_max = (
                    boxslist[pnum][0],
                    boxslist[pnum][1],
                    boxslist[pnum][2],
                    boxslist[pnum][3],
                )
                cv2.putText(
                    annotated_frame,
                    mark,
                    (int((x_min + x_max) / 2), int((y_min + y_max) / 2)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (255, 0, 0),
                    3,
                )

        # writer.write(annotated_frame)
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
