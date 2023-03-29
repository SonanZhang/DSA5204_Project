import torch
import torch.nn as nn
import torch.optim as optim
import os
import cv2

config = [
    (32, 3, 1),
    (64, 3, 2),
    ["B", 1],
    (128, 3, 2),
    ["B", 2],
    (256, 3, 2),
    ["B", 8],
    # first route from the end of the previous block
    (512, 3, 2),
    ["B", 8],
    # second route from the end of the previous block
    (1024, 3, 2),
    ["B", 4],
    # until here is darknet-53
    (512, 1, 1),
    (1024, 3, 1),
    "S",
    (256, 1, 1),
    "U",
    (256, 1, 1),
    (512, 3, 1),
    "S",
    (128, 1, 1),
    "U",
    (128, 1, 1),
    (256, 3, 1),
    "S",
]


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, bias=False, **kwargs),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        return self.conv(x)

class PredictionConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(PredictionConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)

    def forward(self, x):
        return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels, use_residual=True, num_repeats=1):
        super(ResidualBlock, self).__init__()
        self.use_residual = use_residual
        self.layers = nn.ModuleList()
        self.num_repeats = num_repeats
        for _ in range(num_repeats):
            self.layers += [
                nn.Sequential(
                    CNNBlock(channels, channels // 2, kernel_size=1),
                    CNNBlock(channels // 2, channels, kernel_size=3, padding=1)
                )
            ]

    def forward(self, x):
        for layer in self.layers:
            x = layer(x) + self.use_residual * x
        return x


class ScalePrediction(nn.Module):
    def __init__(self, in_channels, num_classes, anchor_per_scale=3):
        super(ScalePrediction, self).__init__()
        self.num_classes = num_classes
        self.anchor_per_scale = anchor_per_scale
        self.layers = nn.ModuleList()
        self.layers = nn.Sequential(
                CNNBlock(in_channels, in_channels * 2, kernel_size=3, padding=1),
                PredictionConvBlock(in_channels * 2, self.anchor_per_scale * (self.num_classes + 5), kernel_size=1)
            )

    def forward(self, x):
        return (
            self.layers(x).reshape(x.shape[0], self.anchor_per_scale, self.num_classes + 5, x.shape[2],
                                   x.shape[3]).permute(0, 1, 3, 4, 2)
        )


class Yolo(nn.Module):
    def __init__(self, in_channels=3, num_classes=20):
        super(Yolo, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.layers = self.create_layers()

    def create_layers(self):
        layers = nn.ModuleList()
        in_channels = self.in_channels

        for layer in config:
            if isinstance(layer, tuple):
                out_channels, kernel_size, stride = layer
                layers.append(
                    CNNBlock(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                             padding=1 if kernel_size == 3 else 0)
                )
                in_channels = out_channels
            elif isinstance(layer, list):
                num_repeats = layer[1]
                layers.append(
                    ResidualBlock(in_channels, num_repeats=num_repeats)
                )
            elif isinstance(layer, str):
                if layer == 'U':
                    layers.append(
                        nn.Upsample(scale_factor=2)
                    )
                    in_channels *= 3
                elif layer == 'S':
                    layers += [
                        ResidualBlock(in_channels, use_residual=False, num_repeats=1),
                        CNNBlock(in_channels, in_channels // 2, kernel_size=1),
                        ScalePrediction(in_channels // 2, num_classes=self.num_classes)
                    ]
                    in_channels //= 2
        return layers

    def forward(self, x):
        outputs = []
        route_connections = []
        for layer in self.layers:
            if isinstance(layer, ScalePrediction):
                outputs.append(layer(x))
                continue
            x = layer(x)
            if isinstance(layer, ResidualBlock) and layer.num_repeats == 8:
                route_connections.append(x)
            elif isinstance(layer, nn.Upsample):
                x = torch.cat([x, route_connections[-1]], dim=1)
                route_connections.pop()
        return outputs

def test():
    num_classes = 20
    model = Yolo(num_classes=num_classes)
    img_size = 416
    x = torch.randn((2, 3, img_size, img_size))
    out = model(x)
    assert out[0].shape == (2, 3, img_size//32, img_size//32, 5 + num_classes)
    assert out[1].shape == (2, 3, img_size//16, img_size//16, 5 + num_classes)
    assert out[2].shape == (2, 3, img_size//8, img_size//8, 5 + num_classes)

test()


