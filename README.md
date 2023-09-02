# FDNET: A Fully Connected Neural Network for Fall Detection

FDNET是一个基于人体关键点数据的摔倒检测的全连接神经网络。它使用yolov8作为人体关键点检测器，然后将人体关键点数据输入一个全连接网络进行分类，判断人物是否摔倒。:question:

## 数据集

本项目使用了[UR Fall Detection Dataset]作为数据集，该数据集包含了40个摔倒和40个日常活动的视频，每个视频都有对应的标签（0表示正常，1表示摔倒）。数据集中的视频都是以25帧每秒的速度拍摄的，分辨率为640x480。:camera:

## 代码文件

本项目包含了以下四个代码文件：

- data.py: 负责将数据集的图片转入yolov8识别并导出后续需要的人体关键点数据以及标签数据。具体来说，它首先将视频分割成图片，并保存在`./data/images`目录下。然后，它使用yolov8对每张图片进行人体关键点检测，并将检测到的17个关键点的坐标以及对应的标签保存在`./data/points`目录下。最后，它将所有图片的关键点数据和标签合并成一个numpy数组，并保存在`./data/data.npy`文件中。:zap:
- model.py: 定义了全连接神经网络的模型。该模型由三层全连接层组成，输入维度为34（17个关键点的x和y坐标），输出维度为2（正常或摔倒）。模型使用了ReLU激活函数和Dropout层防止过拟合。:fire:
```python
class FDNet(nn.Module):
    def __init__(self):
        super(FDNet, self).__init__()
        self.fc1 = nn.Linear(34, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
```
- train.py: 使用之前导出的数据进行了训练。具体来说，它首先加载了`./data/data.npy`文件中的数据，并将其划分为训练集和测试集（比例为8:2）。然后，它创建了一个FDNet模型，并使用Adam优化器和交叉熵损失函数进行训练。训练过程中，它会记录每个epoch的训练损失和测试准确率，并保存在`./logs/train.log`文件中。训练完成后，它会保存最优的模型参数到`./models/fdnet.pth`文件中。:muscle:
- test.py: 使用yolov8识别视频中人体关节点然后在用训练后的网络对这个人的是否摔倒进行检测。具体来说，它首先加载了`./models/fdnet.pth`文件中的模型参数，并创建了一个FDNet模型。然后，它读取了指定的视频文件，并对每一帧进行人体关键点检测和摔倒分类。最后，它将检测和分类的结果显示在视频上，并保存在`./results/result.avi`文件中。:clap:

## 使用方法

要运行本项目，你需要安装以下依赖库：

- torch
- torchvision
- numpy
- opencv-python
- matplotlib

你可以使用以下命令安装这些库：

```bash
pip install -r requirements.txt
```
然后，你可以按照以下步骤执行本项目：

1. 下载[UR Fall Detection Dataset]并解压到`./data/videos`目录下。
2. 运行`python data.py`生成人体关键点数据和标签数据。
3. 运行`python train.py`训练全连接神经网络模型。
4. 运行`python test.py --video ./data/videos/fall-01-cam0.mp4`测试摔倒检测效果。:thumbsup:

## 参考资料

- [UR Fall Detection Dataset]: A public dataset for fall detection from video.
- [yolov8]: An object detection and human pose estimation model based on yolov5.
- [FDNET]: The original GitHub repository of this project.

- : https://www.urfd.org/
- : https://github.com/KayatoDQY/FDNET
