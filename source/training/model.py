import torch
import torch.nn as nn
import torch.nn.functional as F
#from torchsummary import summary
from segmentation_models_pytorch.encoders import get_encoder

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
        
        
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    def forward(self, x):
        return self.up(self.conv(x))


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        
        factor = 2 if bilinear else 1
        
        self.encoder = get_encoder(name = "efficientnet-b4", in_channels = n_channels, depth =5, weights = None)
        
        self.up1 = Up(608, 160 // factor, bilinear)
        self.up2 = Up(136, 56 // factor, bilinear)
        self.up3 = Up(60, 32 // factor, bilinear)
        self.up4 = Up(64, 48, bilinear)
        
        self.outc = OutConv(48, n_classes)
        
        self.ann1_1 = nn.Linear(64512, 512)
        self.ann1_2 = nn.Linear(512, 7)
        #self.ann1_1 = nn.Linear(448, 256)
        #self.ann1_2 = nn.Linear(256, 7)
        
        self.ann2_1 = nn.Linear(64512, 512)
        self.ann2_2 = nn.Linear(512, 1)
        #self.ann2_1 = nn.Linear(448, 256)
        #self.ann2_2 = nn.Linear(256, 1)
        
        #self.maxpool2d = nn.MaxPool2d(8)
        
        self.weights = torch.nn.Parameter(torch.ones(3).float())

    def forward(self, x):

        #encoder
        features = self.encoder(x)
        x0, x1, x2, x3, x4, x5 = features
        
        #decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        output = self.outc(x)
        #print(output.shape)
        
        #a = self.maxpool2d(x5)
        #a = a.view(a.size(0), -1)
        a = x5.view(x5.size(0), -1)
        a1 = self.ann1_1(a)
        a1 = F.relu(a1)
        a1 = self.ann1_2(a1)
        lab = a1
        a2 = self.ann2_1(a)
        a2 = F.relu(a2)        
        a2 = self.ann2_2(a2)
        intensity = a2
        return [output, lab, intensity]

    def get_last_shared_layer(self):
        return self.encoder.get_stages()[-1]

