"""
U-Net model to be used in Multiplaner U-Net.
Reworked from
https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py
"""

from .unet_blocks import *

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, image_size, encoding_size=1000, class_names=None):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.class_names = class_names
        self.encoding_size = encoding_size
        self.image_size = image_size # Assumes square images

        # Hard coded increase of filters by a factor of sqrt(2) for each layer
        self.inc = Double_Conv(n_channels, 90)
        self.down1 = Down(90, 181)
        self.down2 = Down(181, 362)
        self.down3 = Down(362, 724)
        self.down4 = Down(724, 1448)

        self.encoder = Encoder(
            1448,
            self._calculate_fc_in(),
            encoding_size
        )

        self.up1 = Up(1448, 724)
        self.up2 = Up(724, 362)
        self.up3 = Up(362, 181)
        self.up4 = Up(181, 90)
        self.outc = Out_Conv(90, n_classes)

    def forward(self, x, encode=False):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        if encode:
            logits = self.encoder(x)
            return logits

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def _calculate_fc_in(self):
        bottleneck_imsize = self.image_size // 16 # 4 max_pool
        fc_imsize = bottleneck_imsize // 4 # 2 max_pool
        fc_feature_channels = 1448 // 4
        fc_in = fc_imsize * fc_feature_channels
        return fc_in