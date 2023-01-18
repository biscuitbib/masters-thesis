from torch import nn

class Encoder(nn.Module):
    def __init__(self, in_channels, fc_in, vector_size):
        super().__init__()
        self.in_channels = in_channels
        self.fc_in = fc_in
        self.vector_size = vector_size
        self.encoder = nn.Sequential(
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(
                in_channels,
                in_channels//2,
                kernel_size=3,
                padding="same"
            ),
            nn.ReLU(),
            #nn.BatchNorm2d(in_channels//2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(
                in_channels//2,
                in_channels//4,
                kernel_size=3,
                padding="same"
            ),
            nn.ReLU(),
            #nn.BatchNorm2d(in_channels//4),
            nn.Flatten(),
            nn.Linear(
                fc_in,
                vector_size
            )
        )

    def forward(self, x):
        x = self.encoder(x)
        return x