import torch.nn as nn

class CNN_model(nn.Module):
    def __init__(self):
        super(CNN_model, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 11, 3, 3),
            # 82 * 82 * 16
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 41 * 41 * 16
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 2, 1),
            # 20 * 20 * 32
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 10 * 10 * 32
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 5, 1, 1),
            # 8 * 8 * 64
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 4 * 4 * 64
        )
        self.out = nn.Linear(4*4*64, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output