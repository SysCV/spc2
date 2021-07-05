import torch.nn as nn
import torch.nn.functional as F
from utils.util import weights_init


class end_layer(nn.Module):
    def __init__(self, args, in_channels, out_dim, activate=None):
        super(end_layer, self).__init__()
        self.args, self.activate = args, activate

        self.conv1 = nn.Conv2d(in_channels, 16, 5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 32, 1, stride=1, padding=0)

        self.flatten_len = int(32 * (self.args.frame_width / 128) * (self.args.frame_height / 128))
        self.fc1 = nn.Linear(self.flatten_len, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, out_dim)

        self.apply(weights_init)

    def forward(self, x):
        # print("----")
        # print(x.shape)
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2, stride=2), inplace=True)
        # print(x.shape)
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2, stride=2), inplace=True)
        # print(x.shape)
        x = F.relu(F.max_pool2d(self.conv3(x), kernel_size=2, stride=2), inplace=True)
        # print(x.shape)
        x = F.relu(F.max_pool2d(self.conv4(x), kernel_size=2, stride=2), inplace=True)
        # print(x.shape)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        # print("----")
        x = F.relu(self.fc11(x), inplace=True)
        x = F.relu(self.fc22(x), inplace=True)
        x = self.fc33(x)

        if self.activate is not None:
            x = self.activate(x)

        return x
