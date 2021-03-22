import torch.nn as nn
import torch.nn.functional as F

class DensityNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1792, out_channels=1024, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(1024 * 5 * 4, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 1)

    def forward(self, x):
        out = F.max_pool2d(nn.ReLU(self.conv1(x)),2)
        out = out.view(-1, 1024 * 4 * 5)
        out = nn.ReLU(self.fc1(out))
        out = nn.ReLU(self.fc2(out))
        out = nn.ReLU(self.fc3(out))
        return out