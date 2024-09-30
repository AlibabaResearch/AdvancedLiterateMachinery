import torch
import torch.nn as nn
import torch.nn.functional as F


class SEResBasicBlock(nn.Module):
    def __init__(self, planes):
        super(SEResBasicBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(planes, planes, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(planes, planes, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        res = self.conv1(x)
        res = self.conv2(res)
        res = x + res

        return res


class SEResBlock(nn.Module):
    def __init__(self, inplanes, planes, layers):
        super(SEResBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=(1, 1), padding=0),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(planes, planes, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
        )

        Sequential = []
        for idx in range(layers - 1):
            Sequential.append(SEResBasicBlock(planes))

        self.SERes = nn.Sequential(*Sequential)

    def forward(self, x):
        res_input = self.conv1(x)
        res = self.conv2(x)
        res = self.conv3(res)
        res = res_input + res

        res = self.SERes(res)
        return res


class FCNEncoder(nn.Module):
    def __init__(self):
        super(FCNEncoder, self).__init__()
        self.conv0_0 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.conv0_1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.p1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0))
        self.SERes1 = SEResBlock(64, 64, 1)
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.p2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0))
        self.SERes2 = SEResBlock(64, 96, 2)
        self.conv2_1 = nn.Sequential(
            nn.Conv2d(96, 96, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
        )

        self.p3 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1), padding=(0, 0))
        self.SERes3 = SEResBlock(96, 192, 4)
        self.conv3_1 = nn.Sequential(
            nn.Conv2d(192, 192, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
        )

        self.p4 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1), padding=(0, 0))
        self.SERes4 = SEResBlock(192, 384, 4)
        self.conv4_1 = nn.Sequential(
            nn.Conv2d(384, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.p5 = nn.MaxPool2d(kernel_size=(4, 1), stride=(4, 1), padding=(0, 0))
        self.conv5_1 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv0_0(x)
        x = self.conv0_1(x)

        x = self.p1(x)
        x = self.SERes1(x)
        x = self.conv1_1(x)

        x = self.p2(x)
        x = self.SERes2(x)
        x = self.conv2_1(x)

        x = self.p3(x)
        x = self.SERes3(x)
        x = self.conv3_1(x)

        x = self.p4(x)
        x = self.SERes4(x)
        x = self.conv4_1(x)

        x = self.p5(x)
        x = self.conv5_1(x)

        return x
