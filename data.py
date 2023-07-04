from ultralytics import YOLO
import torch
import cv2
import os
import math


def elu_distance(a, b):
    x = sum(pow((a_ - b_), 2) for a_, b_ in zip(a, b))
    return math.sqrt(x)


yolomodel = YOLO("model/yolo/yolov8x-pose.pt")
datafile = "data_raw/"
datasave = "data/"
stepfilelist = os.listdir(datafile)
for stepfile in stepfilelist:
    stepfile = stepfile + "/"
    datanumlist = os.listdir(datafile + stepfile)
    for datanum in datanumlist:
        datanum = datanum + "/"
        filenamelist = os.listdir(datafile + stepfile + datanum)
        for filename in filenamelist:
            if os.path.splitext(filename)[1] == ".png":
                dataname = os.path.splitext(filename)[0]
                imgname = datafile + stepfile + datanum + dataname + ".png"
                txtname = datafile + stepfile + datanum + dataname + ".txt"
                print("imgname ", imgname)
                with open(txtname, "r") as f:
                    lines = f.readlines()
                    print("person num ", len(lines))
                    if len(lines) != 0:
                        results = yolomodel(imgname)

                        resultboxs = []
                        for result in results:
                            if len(result.boxes.xyxy.tolist()) != 0:
                                resultbox = result.boxes.xyxy.tolist()[0]
                                resultboxs.append(resultbox)

                        for line in lines:
                            line = line.split()
                            line = [int(num) for num in line]
                            box = line[1:4]
                            classFD = line[0]

                            minelu = 9999999999
                            minmark = -1
                            leni = 0
                            for resultbox in resultboxs:
                                elu = elu_distance(resultbox, box)
                                if minelu > elu:
                                    minmark = leni
                                leni = leni + 1

                            if minmark != -1 and results[
                                minmark
                            ].keypoints.xyn.shape == torch.Size([1, 17, 2]):
                                print(
                                    "save keypoint ",
                                    datasave
                                    + stepfile
                                    + datanum
                                    + dataname
                                    + "_"
                                    + str(minmark)
                                    + "_keypoint"
                                    + ".pt",
                                )
                                print(
                                    "save label ",
                                    datasave
                                    + stepfile
                                    + datanum
                                    + dataname
                                    + "_"
                                    + str(minmark)
                                    + "_label"
                                    + ".pt",
                                )
                                keypoint = results[minmark].keypoints.xyn.reshape(
                                    17 * 2
                                )
                                torch.save(
                                    keypoint,
                                    datasave
                                    + stepfile
                                    + datanum
                                    + dataname
                                    + "_"
                                    + str(minmark)
                                    + "_keypoint"
                                    + ".pt",
                                )
                                torch.save(
                                    torch.tensor([classFD]).unsqueeze(0),
                                    datasave
                                    + stepfile
                                    + datanum
                                    + dataname
                                    + "_"
                                    + str(minmark)
                                    + "_label"
                                    + ".pt",
                                )

                # img=cv2.imread(imgname)
                # cv2.imshow("img",img)
                # cv2.waitKey(2)
