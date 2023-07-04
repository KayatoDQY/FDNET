from ultralytics import YOLO
import os
import cv2
import torch
from torch import nn
from torch import optim

from model import FDNET

numepoch = 100
net = FDNET()
net = net.to(torch.device("cuda"))
opt = optim.Adam(net.parameters(), lr=1e-4)
crossentropyloss = nn.CrossEntropyLoss()

# model = YOLO('model/yolo/yolov8n-pose.pt')
# results = model('data_raw/test/split4/split4_012.png')
# net=FDNET().cuda()
# resFD=net.forward(results[0].keypoints.xyn.reshape(17*2))
# print(resFD.shape)
# print(resFD)
# print(results[0].keypoints.xyn.shape == torch.Size([1, 17, 2]))

# input=torch.load("data/test/split4/split4_011_0_keypoint.pt")
# label=torch.load("data/test/split4/split4_011_0_label.pt")

# input=input.unsqueeze(0).cuda()
# if label==torch.tensor([[1]]):
#    label=torch.tensor([1]).cuda()
# elif label==torch.tensor([[-1]]):
#    label=torch.tensor([2]).cuda()

# output=net.forward(input)
# print(output)
# print(output.shape,label.shape)
# opt = optim.Adam(net.parameters(), lr=3e-4)
# loss_func = nn.NLLLoss(reduction="sum")

# loss=loss_func(output,label)
# print(loss)
# opt.zero_grad()
# loss.backward()
# opt.step()

for epoch in range(numepoch):
    for datanum in os.listdir("data/" + "train" + "/"):
        for data in os.listdir("data/" + "train" + "/" + datanum + "/"):
            datastrlist = data.split("_")
            if datastrlist[3] == "keypoint.pt":
                input = torch.load("data/" + "train" + "/" + datanum + "/" + data)
                labelname = "_".join(datastrlist[0:3]) + "_label.pt"
                label = torch.load("data/" + "train" + "/" + datanum + "/" + labelname)

                input = input.unsqueeze(0).cuda()
                if label == torch.tensor([[1]]):
                    label = torch.tensor([[1, 0]], dtype=torch.float).cuda()
                elif label == torch.tensor([[-1]]):
                    label = torch.tensor([[0, 1]], dtype=torch.float).cuda()

                output = net.forward(input)

                loss = crossentropyloss(output, label)
                opt.zero_grad()
                loss.backward()
                opt.step()
    loss_sum = 0
    sumall = 0
    for datanum in os.listdir("data/" + "valid" + "/"):
        for data in os.listdir("data/" + "valid" + "/" + datanum + "/"):
            datastrlist = data.split("_")
            if datastrlist[3] == "keypoint.pt":
                input = torch.load("data/" + "valid" + "/" + datanum + "/" + data)
                labelname = "_".join(datastrlist[0:3]) + "_label.pt"
                label = torch.load("data/" + "valid" + "/" + datanum + "/" + labelname)

                input = input.unsqueeze(0).cuda()
                if label == torch.tensor([[1]]):
                    label = torch.tensor([[1, 0]], dtype=torch.float).cuda()
                elif label == torch.tensor([[-1]]):
                    label = torch.tensor([[0, 1]], dtype=torch.float).cuda()

                output = net.forward(input)
                loss = crossentropyloss(output, label)
                loss_sum = loss_sum + loss
                sumall = sumall + 1

    print("epoch:  %d ,loss  %.6f " % (epoch, loss_sum / sumall))

torch.save(net.state_dict(), "model/FDNET.pt")
