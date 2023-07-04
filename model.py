import torch
from torch import nn
import torch.nn.functional as F


class FDNET(nn.Module):
    def __init__(self):
        super(FDNET, self).__init__()
        self.fc1 = nn.Linear(17 * 2, 17 * 8)
        self.fc2 = nn.Linear(17 * 8, 17 * 4)
        self.fc3 = nn.Linear(17 * 4, 17)
        self.fc4 = nn.Linear(17, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)
