# name: Cao Yintao
# date: 2024/1/3 , 15:28
# fcn_res101.py
import torch
import torch.nn as nn
import torchvision.models as models
import os
script_name = os.path.basename(__file__)
# 解码器部分
class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

# 完整的Res-SegNet模型
class ResSegNet(nn.Module):
    def __init__(self, n_classes):
        super(ResSegNet, self).__init__()
        # 使用预训练的ResNet-101模型作为编码器的骨干
        backbone = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
        self.encoder = nn.Sequential(*list(backbone.children())[:-2])

        # 解码器部分
        self.decoder1 = Decoder(2048, 1024)
        self.decoder2 = Decoder(1024, 512)
        self.decoder3 = Decoder(512, 256)
        self.decoder4 = Decoder(256, 64)
        self.decoder5 = Decoder(64, 64)

        # 分类头
        self.classifier = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        # 编码
        x = self.encoder(x)

        # 解码
        x = self.decoder1(x)
        x = self.decoder2(x)
        x = self.decoder3(x)
        x = self.decoder4(x)
        x = self.decoder5(x)

        # 分类
        x = self.classifier(x)

        return x

if __name__ == "__main__":
    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img = torch.rand(1, 3, 256, 256)

    net = ResSegNet(9)
    output = net(img)
    print('%s, training this file'%(script_name))

    # 将网络拷贝到deivce中
    net.to(device=device)
    print(output.shape)
    print(output)
    print(type(output))