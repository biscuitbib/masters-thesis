import numpy as np

import torch
import torch.nn as nn

class not_Unet(nn.Module):
    """
    U-net for use in multiplanar u-net
    """
    def __init__(self,
                 n_classes,
                 img_rows=None,
                 img_cols=None,
                 n_channels=1,
                 depth=4,
                 cf=np.sqrt(2)):
        super(not_Unet, self).__init__()
        self.n_classes = n_classes
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.n_channels = n_channels
        self.depth = depth
        self.cf = cf

        # Initial number of filters which will double/half after every down/up layer
        self.init_filters = 64

    def _run_contractive_path(self, _x):
        x = _x
        cross_connections = []
        in_channels = self.n_channels
        filters = self.init_filters
        for i in range(self.depth):
            out_channels = int(filters * self.cf)

            tmp_out = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    padding="same",
                ),
                nn.ReLU(),
                nn.Conv2d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    padding="same",
                ),
                nn.ReLU(),
                nn.BatchNorm2d(out_channels)
            )(x)

            x = nn.MaxPool2d(kernel_size=2)(tmp_out)

            cross_connections.append(tmp_out)

            in_channels = out_channels
            filters *= 2

        return x, cross_connections, filters

    def _run_encoder(self, _x, filters):
        in_channels = int(filters / 2 * self.cf)
        out_channels = int(filters * self.cf)

        x = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding="same",
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding="same",
            ),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )(_x)

        return x

    def _run_expansive_path(self, _x, filters, cross_connections):
        x = _x
        in_channels = int(filters * self.cf)
        filters = filters // 2
        for i in range(self.depth):
            out_channels = int(filters * self.cf)

            bn_out = nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=2,
                    padding="same"
                ),
                nn.ReLU(),
                nn.BatchNorm2d(out_channels)
            )(x)

            bn_cross = cross_connections[i]

            merge_bn = torch.cat([bn_cross, bn_out], dim=1)

            x = nn.Sequential(
                nn.Conv2d(
                    in_channels=out_channels * 2,
                    out_channels=out_channels,
                    kernel_size=3,
                    padding="same"
                ),
                nn.ReLU(),
                nn.Conv2d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    padding="same"
                ),
                nn.ReLU(),
                nn.BatchNorm2d(out_channels)
            )(merge_bn)

            in_channels = out_channels
            filters /= 2

        return x

    def forward(self, x):
        x, cross_connections, filters = self._run_contractive_path(x)
        x = self._run_encoder(x, filters)
        x = self._run_expansive_path(x, filters, cross_connections[::-1])
        x = nn.Conv2d(
            in_channels=int(self.init_filters * self.cf),
            out_channels=self.n_classes,
            kernel_size=1
        )(x)
        x = nn.Softmax(dim=1)(x)
        return x

class Unet(nn.Module):
    def contracting_block(self, in_channels, out_channels, kernel_size=3):
        """
        This function creates one contracting block
        """
        block = torch.nn.Sequential(
                    torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm2d(out_channels),
                    torch.nn.Conv2d(kernel_size=kernel_size, in_channels=out_channels, out_channels=out_channels),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm2d(out_channels),
                )
        return block

    def expansive_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
        """
        This function creates one expansive block
        """
        block = torch.nn.Sequential(
                torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel),
                torch.nn.ReLU(),
                torch.nn.BatchNorm2d(mid_channel),
                torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=mid_channel),
                torch.nn.ReLU(),
                torch.nn.BatchNorm2d(mid_channel),
                torch.nn.ConvTranspose2d(in_channels=mid_channel, out_channels=out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
                )
        return  block

    def final_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
        """
        This returns final block
        """
        block = torch.nn.Sequential(
                torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel),
                torch.nn.ReLU(),
                torch.nn.BatchNorm2d(mid_channel),
                torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=mid_channel),
                torch.nn.ReLU(),
                torch.nn.BatchNorm2d(mid_channel),
                torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=out_channels, padding=1),
                torch.nn.ReLU(),
                torch.nn.BatchNorm2d(out_channels),
                )
        return  block

    def __init__(self, in_channel, out_channel):
        super(Unet, self).__init__()
        #Encode
        self.conv_encode1 = self.contracting_block(in_channels=in_channel, out_channels=64)
        self.conv_maxpool1 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv_encode2 = self.contracting_block(64, 128)
        self.conv_maxpool2 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv_encode3 = self.contracting_block(128, 256)
        self.conv_maxpool3 = torch.nn.MaxPool2d(kernel_size=2)
        # Bottleneck
        self.bottleneck = torch.nn.Sequential(
                            torch.nn.Conv2d(kernel_size=3, in_channels=256, out_channels=512),
                            torch.nn.ReLU(),
                            torch.nn.BatchNorm2d(512),
                            torch.nn.Conv2d(kernel_size=3, in_channels=512, out_channels=512),
                            torch.nn.ReLU(),
                            torch.nn.BatchNorm2d(512),
                            torch.nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1, output_padding=1)
                            )
        # Decode
        self.conv_decode3 = self.expansive_block(512, 256, 128)
        self.conv_decode2 = self.expansive_block(256, 128, 64)
        self.final_layer = self.final_block(128, 64, out_channel)

    def crop_and_concat(self, upsampled, bypass, crop=False):
        """
        This layer crop the layer from contraction block and concat it with expansive block vector
        """
        if crop:
            c = (bypass.size()[2] - upsampled.size()[2]) // 2
            bypass = F.pad(bypass, (-c, -c, -c, -c))
        return torch.cat((upsampled, bypass), 1)

    def forward(self, x):
        # Encode
        encode_block1 = self.conv_encode1(x)
        encode_pool1 = self.conv_maxpool1(encode_block1)
        encode_block2 = self.conv_encode2(encode_pool1)
        encode_pool2 = self.conv_maxpool2(encode_block2)
        encode_block3 = self.conv_encode3(encode_pool2)
        encode_pool3 = self.conv_maxpool3(encode_block3)
        # Bottleneck
        bottleneck1 = self.bottleneck(encode_pool3)
        # Decode
        decode_block3 = self.crop_and_concat(bottleneck1, encode_block3, crop=True)
        cat_layer2 = self.conv_decode3(decode_block3)
        decode_block2 = self.crop_and_concat(cat_layer2, encode_block2, crop=True)
        cat_layer1 = self.conv_decode2(decode_block2)
        decode_block1 = self.crop_and_concat(cat_layer1, encode_block1, crop=True)
        final_layer = self.final_layer(decode_block1)
        return  final_layer