"""
U-Net model to be used in Multiplaner U-Net.
Reworked from
https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py
"""
from .unet_blocks import *


class UNet(nn.Module):
    """
    U-net used in the multiplanar U-net.
    """
    def __init__(self, n_channels, n_classes, image_size, class_names=None, encode=False):
        """
        Parameters:
            n_channels: Number of channels in the input image, e.g. for RGB images n_channels = 3
            n_classes: Number of classes including background, e.g. for binary classification, n_classes = 2
            image_size: The size of the image slice. Assumes square images, e.g. for 150 X 150 images, image_size=150.
            encoding_size: The size of the vector returned by the bottleneck encoder.
            class_names: Optional names for the n_classes - 1 foreground classes.
            encode: Whether to return segmentation masks or encoded bottleneck vectors.
        """
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.class_names = class_names
        self.image_size = image_size # Assumes square images
        self.encode = encode
        self.fc_in = self._calculate_fc_in() # Flattened size of bottleneck

        # Hard coded increase of filters by a factor of sqrt(2) for each layer
        self.inc = Double_Conv(n_channels, 90)
        self.down1 = Down(90, 181)
        self.down2 = Down(181, 362)
        self.down3 = Down(362, 724)
        self.down4 = Down(724, 1448)

        self.up1 = Up(1448, 724)
        self.up2 = Up(724, 362)
        self.up3 = Up(362, 181)
        self.up4 = Up(181, 90)
        self.outc = Out_Conv(90, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        if self.encode:
            if torch.any(torch.isnan(x5)):
                raise Exception("U-net produces NaN")
            return x5

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
        fc_in = fc_imsize * fc_imsize * fc_feature_channels
        return fc_in