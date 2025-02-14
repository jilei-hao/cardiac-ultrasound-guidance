

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class ResNet34UNet(nn.Module):
    def __init__(self, num_landmarks=42, pretrained=False):
        super().__init__()
        
        resnet = torchvision.models.resnet34(weights=None)

        self.encoder0 = nn.Sequential(
            resnet.conv1, 
            resnet.bn1,
            resnet.relu
        )
        self.pool0 = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        # A bigger decoder (512->512->256->128->64->64)
        self.up4 = self._up_block(512, 512)
        self.up3 = self._up_block(512 + 256, 256)
        self.up2 = self._up_block(256 + 128, 128)
        self.up1 = self._up_block(128 + 64, 64)
        self.up0 = self._up_block(64 + 64, 64)

        self.final_conv = nn.Conv2d(64, num_landmarks, kernel_size=1)

    def _up_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Upsample from (112,112)->(128,128) so the strides are nice multiples of 2
        x = F.interpolate(x, size=(128,128), mode='bilinear', align_corners=False)

        e0 = self.encoder0(x)    # => (B,64,64,64)
        p0 = self.pool0(e0)      # => (B,64,32,32)
        e1 = self.encoder1(p0)   # => (B,64,32,32)
        e2 = self.encoder2(e1)   # => (B,128,16,16)
        e3 = self.encoder3(e2)   # => (B,256,8,8)
        e4 = self.encoder4(e3)   # => (B,512,4,4)

        d4 = self.up4(e4)        # => (B,512,8,8)
        d4 = torch.cat([d4, e3], dim=1)  # => (B,768,8,8)

        d3 = self.up3(d4)        # => (B,256,16,16)
        d3 = torch.cat([d3, e2], dim=1)  # => (B,384,16,16)

        d2 = self.up2(d3)        # => (B,128,32,32)
        d2 = torch.cat([d2, e1], dim=1)  # => (B,192,32,32)

        d1 = self.up1(d2)        # => (B,64,64,64)
        d1 = torch.cat([d1, e0], dim=1)  # => (B,128,64,64)

        d0 = self.up0(d1)        # => (B,64,128,128)
        out = self.final_conv(d0)# => (B,42,128,128)

        # Downsample back to (112,112)
        out = F.interpolate(out, size=(112,112), mode='bilinear', align_corners=False)
        return out
