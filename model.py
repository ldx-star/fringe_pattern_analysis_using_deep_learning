import torch.nn as nn
import torch.nn.functional as F
import torch


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution without padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.conv2 = conv3x3(planes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        if stride == 1:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(
                conv1x1(in_planes, planes, stride=stride),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        y = x
        y = self.relu(self.bn1(self.conv1(y)))
        y = self.bn2(self.conv2(y))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x + y)


class ResNetFPN(nn.Module):
    def __init__(self):
        super().__init__()
        # Config
        block = BasicBlock
        initial_dim = 128
        block_dims = [128, 196, 256]

        self.conv_128_1 = nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_128_2 = nn.Conv2d(128, 2, kernel_size=3, stride=1, padding=1, bias=False)

        # cnn1
        # Class Variable
        self.in_planes = initial_dim

        # Networks
        self.conv1_1 = nn.Conv2d(1, initial_dim, kernel_size=7, stride=2, padding=3, bias=False)

        self.bn1_1 = nn.BatchNorm2d(initial_dim)
        self.relu_1 = nn.ReLU(inplace=True)

        self.layer1_1 = self._make_layer(block, block_dims[0], stride=1)  # 1/2
        self.layer2_1 = self._make_layer(block, block_dims[1], stride=2)  # 1/4
        self.layer3_1 = self._make_layer(block, block_dims[2], stride=2)  # 1/8

        # 3. FPN upsample
        self.layer3_outconv_1 = conv1x1(block_dims[2], block_dims[2])
        self.layer2_outconv_1 = conv1x1(block_dims[1], block_dims[2])
        self.layer2_outconv2_1 = nn.Sequential(
            conv3x3(block_dims[2], block_dims[2]),
            nn.BatchNorm2d(block_dims[2]),
            nn.LeakyReLU(),
            conv3x3(block_dims[2], block_dims[1]),
        )
        self.layer1_outconv_1 = conv1x1(block_dims[0], block_dims[1])
        self.layer1_outconv2_1 = nn.Sequential(
            conv3x3(block_dims[1], block_dims[1]),
            nn.BatchNorm2d(block_dims[1]),
            nn.LeakyReLU(),
            conv3x3(block_dims[1], block_dims[0]),
        )

        # cnn2
        # Class Variable
        # Networks
        self.in_planes = initial_dim

        self.conv1_2 = nn.Conv2d(2, initial_dim, kernel_size=7, stride=2, padding=3, bias=False)

        self.bn1_2 = nn.BatchNorm2d(initial_dim)
        self.relu_2 = nn.ReLU(inplace=True)

        self.layer1_2 = self._make_layer(block, block_dims[0], stride=1)  # 1/2
        self.layer2_2 = self._make_layer(block, block_dims[1], stride=2)  # 1/4
        self.layer3_2 = self._make_layer(block, block_dims[2], stride=2)  # 1/8

        # 3. FPN upsample
        self.layer3_outconv_2 = conv1x1(block_dims[2], block_dims[2])
        self.layer2_outconv_2 = conv1x1(block_dims[1], block_dims[2])
        self.layer2_outconv2_2 = nn.Sequential(
            conv3x3(block_dims[2], block_dims[2]),
            nn.BatchNorm2d(block_dims[2]),
            nn.LeakyReLU(),
            conv3x3(block_dims[2], block_dims[1]),
        )
        self.layer1_outconv_2 = conv1x1(block_dims[0], block_dims[1])
        self.layer1_outconv2_2 = nn.Sequential(
            conv3x3(block_dims[1], block_dims[1]),
            nn.BatchNorm2d(block_dims[1]),
            nn.LeakyReLU(),
            conv3x3(block_dims[1], block_dims[0]),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, dim, stride=1):
        layer1 = block(self.in_planes, dim, stride=stride)
        layer2 = block(dim, dim, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, input_img):

        # CNN1
        # ResNet Backbone
        x0 = self.relu_1(self.bn1_1(self.conv1_1(input_img)))
        x1 = self.layer1_1(x0)  # 1/2
        x2 = self.layer2_1(x1)  # 1/4
        x3 = self.layer3_1(x2)  # 1/8

        # FPN
        x3_out = self.layer3_outconv_1(x3)

        x3_out_2x = F.interpolate(x3_out, scale_factor=2., mode='bilinear', align_corners=True)
        x2_out = self.layer2_outconv_1(x2)
        x2_out = self.layer2_outconv2_1(x2_out + x3_out_2x)

        x2_out_1x = F.interpolate(x2_out, scale_factor=2., mode='bilinear', align_corners=True)
        x1_out = self.layer1_outconv_1(x1)
        x1_out = self.layer1_outconv2_1(x1_out + x2_out_1x)

        cnn1_out = self.conv_128_1(F.interpolate(x1_out, scale_factor=2., mode='bilinear', align_corners=True))

        # CNN2
        input2 = torch.concat((input_img, cnn1_out), 1)
        x0_2 = self.relu_2(self.bn1_2(self.conv1_2(input2)))
        x1_2 = self.layer1_2(x0_2)  # 1/2
        x2_2 = self.layer2_2(x1_2)  # 1/4
        x3_2 = self.layer3_2(x2)  # 1/8

        # # FPN
        x3_out_2 = self.layer3_outconv_2(x3_2)

        x3_out_2x_2 = F.interpolate(x3_out_2, scale_factor=2., mode='bilinear', align_corners=True)
        x2_out_2 = self.layer2_outconv_2(x2_2)
        x2_out_2 = self.layer2_outconv2_2(x2_out_2 + x3_out_2x_2)

        x2_out_1x_2 = F.interpolate(x2_out_2, scale_factor=2., mode='bilinear', align_corners=True)
        x1_out_2 = self.layer1_outconv_2(x1_2)
        x1_out_2 = self.layer1_outconv2_2(x1_out_2 + x2_out_1x_2)

        cnn2_out_2 = self.conv_128_2(F.interpolate(x1_out_2, scale_factor=2., mode='bilinear', align_corners=True))

        return cnn1_out, cnn2_out_2
