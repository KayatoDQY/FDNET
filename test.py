from ultralytics import YOLO
import os
import cv2
import torch
from torch import nn
from torch import optim

from model import FDNET


net = FDNET()
net = net.to(torch.device("cuda"))
m_state_dict = torch.load("model/FDNET.pt")
net.load_state_dict(m_state_dict)

for datanum in os.listdir("data/" + "test" + "/"):
    rightnum = 0
    falsenum = 0
    for data in os.listdir("data/" + "test" + "/" + datanum + "/"):
        datastrlist = data.split("_")
        if os.path.splitext(data)[1] == ".pt":
            if datastrlist[3] == "keypoint.pt":
                input = torch.load("data/" + "test" + "/" + datanum + "/" + data)
                labelname = "_".join(datastrlist[0:3]) + "_label.pt"
                label_ = torch.load("data/" + "test" + "/" + datanum + "/" + labelname)

                input = input.unsqueeze(0).cuda()
                if label_ == torch.tensor([[1]]):
                    label = torch.tensor([[1, 0]], dtype=torch.float).cuda()
                elif label_ == torch.tensor([[-1]]):
                    label = torch.tensor([[0, 1]], dtype=torch.float).cuda()

                output = net.forward(input)
                output = output.tolist()
                if output[0][0] > output[0][1] and label_ == torch.tensor([[1]]):
                    rightnum = rightnum + 1
                elif output[0][0] < output[0][1] and label_ == torch.tensor([[-1]]):
                    rightnum = rightnum + 1
                else:
                    falsenum = falsenum + 1
                    print(
                        "data/"
                        + "test"
                        + "/"
                        + datanum
                        + "/"
                        + "_".join(datastrlist[0:3])
                    )

    print(
        "true:  %d ,false  %d ,   百分之%.2f"
        % (rightnum, falsenum, rightnum / (rightnum + falsenum) * 100)
    )
