import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings(action="ignore")


def weights_init(init_type="kaiming", gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (
            classname.find("Conv") != -1 or classname.find("Linear") != -1
        ):
            if init_type == "normal":
                nn.init.normal_(m.weight.data, 0.0, gain)
            elif init_type == "xavier":
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == "kaiming":
                nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                nn.init.orthogonal_(m.weight.data, gain=gain)

            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)

        elif classname.find("BatchNorm2d") != -1:
            nn.init.normal_(m.weight.data, 1.0, gain)
            nn.init.constant_(m.bias.data, 0.0)

    return init_func


class DoubleConv(nn.Module):
    """(conv => BN => ReLU) * 2"""

    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class InConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(InConv, self).__init__()
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down, self).__init__()
        self.mpconv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))

    def forward(self, x):
        x = self.mpconv(x)
        return x


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(Up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class ResidualTanh(nn.Module):
    """Activation: 0.5 + scale * tanh(x)

    Output is centered at 0.5 (blend-map neutral) with a controllable range:
        scale=0.50  →  [0.0,  1.0]   (full range, equivalent to sigmoid-like)
        scale=0.25  →  [0.25, 0.75]
        scale=0.20  →  [0.30, 0.70]   (good default for subtle chin retouching)

    Advantages over sigmoid for blend-map prediction:
    - Explicitly centered at neutral: tanh(0)=0 → output=0.5
    - Constrainable range prevents extreme blend values
    - Symmetric gradients around the operating point
    """

    def __init__(self, scale: float = 0.5):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return 0.5 + self.scale * torch.tanh(x)

    def extra_repr(self):
        return f"scale={self.scale}, range=[{0.5 - self.scale:.2f}, {0.5 + self.scale:.2f}]"


class UNet(nn.Module):
    def __init__(
        self, n_channels, n_classes, deep_supervision=False, init_weights=True
    ):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.inc = InConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, n_classes)

        self.dsoutc4 = OutConv(256, n_classes)
        self.dsoutc3 = OutConv(128, n_classes)
        self.dsoutc2 = OutConv(64, n_classes)
        self.dsoutc1 = OutConv(64, n_classes)

        self.sigmoid = nn.Sigmoid()

        if init_weights:
            self.apply(weights_init())

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x44 = self.up1(x5, x4)
        x33 = self.up2(x44, x3)
        x22 = self.up3(x33, x2)
        x11 = self.up4(x22, x1)
        x0 = self.outc(x11)
        x0 = self.sigmoid(x0)
        return x0


class BaseUNetHalf(nn.Module):
    def __init__(
        self, n_channels, n_classes, deep_supervision=False, init_weights=True,
        last_layer_activation="sigmoid", blend_scale=0.5,
    ):
        super(BaseUNetHalf, self).__init__()
        self.deep_supervision = deep_supervision

        self.inc = InConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)

        self.down3 = Down(256, 256)
        self.down4 = Down(256, 256)
        self.up1 = Up(512, 256)  # 256 + 256 = 512 input channels

        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, n_classes)

        # Deep supervision outputs (only used when deep_supervision=True)
        if deep_supervision:
            self.dsoutc4 = OutConv(256, n_classes)
            self.dsoutc3 = OutConv(128, n_classes)
            self.dsoutc2 = OutConv(64, n_classes)
            self.dsoutc1 = OutConv(64, n_classes)

        if last_layer_activation == "sigmoid":
            self.last_layer_activation = nn.Sigmoid()
        elif last_layer_activation == "tanh":
            self.last_layer_activation = nn.Tanh()
        elif last_layer_activation == "residual_tanh":
            self.last_layer_activation = ResidualTanh(scale=blend_scale)
        else:
            raise ValueError(f"Invalid last layer activation: {last_layer_activation}")

        if init_weights:
            self.apply(weights_init())

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x44 = self.up1(x5, x4)
        x33 = self.up2(x44, x3)
        x22 = self.up3(x33, x2)
        x11 = self.up4(x22, x1)
        x0 = self.outc(x11)
        x0 = self.last_layer_activation(x0)
        return x0


class BaseUNetHalfLite(nn.Module):
    """
    Lighter variant of BaseUNetHalf that bilinear-downscales the input to half
    resolution before encoding and bilinear-upscales the output back to the
    original resolution.  This removes one encoder + decoder level compared to
    BaseUNetHalf, cutting parameters roughly in half and reducing peak VRAM
    significantly (the highest-resolution feature maps are 512 instead of 1024).

    Particularly well-suited for blend-map prediction, where the output is
    inherently smooth and doesn't need pixel-level high-frequency detail.

    Spatial progression (for 1024 input):
        1024 ─bilinear→ 512 ─inc→ 512 ─d1→ 256 ─d2→ 128 ─d3→ 64 (bottleneck)
        64 ─u1→ 128 ─u2→ 256 ─u3→ 512 ─outc→ 512 ─bilinear→ 1024
    """

    def __init__(
        self, n_channels, n_classes, deep_supervision=False, init_weights=True,
        last_layer_activation="sigmoid", blend_scale=0.5,
    ):
        super(BaseUNetHalfLite, self).__init__()
        self.deep_supervision = deep_supervision

        # Encoder  (internal resolution starts at input_size / 2)
        self.inc = InConv(n_channels, 64)    # 512
        self.down1 = Down(64, 128)           # 256
        self.down2 = Down(128, 256)          # 128
        self.down3 = Down(256, 256)          # 64  (bottleneck)

        # Decoder
        self.up1 = Up(512, 128)   # 256+256 → 128  @ 128
        self.up2 = Up(256, 64)    # 128+128 → 64   @ 256
        self.up3 = Up(128, 64)    # 64+64   → 64   @ 512
        self.outc = OutConv(64, n_classes)

        # Deep supervision outputs (only used when deep_supervision=True)
        if deep_supervision:
            self.dsoutc3 = OutConv(128, n_classes)
            self.dsoutc2 = OutConv(64, n_classes)
            self.dsoutc1 = OutConv(64, n_classes)

        if last_layer_activation == "sigmoid":
            self.last_layer_activation = nn.Sigmoid()
        elif last_layer_activation == "tanh":
            self.last_layer_activation = nn.Tanh()
        elif last_layer_activation == "residual_tanh":
            self.last_layer_activation = ResidualTanh(scale=blend_scale)
        else:
            raise ValueError(f"Invalid last layer activation: {last_layer_activation}")

        if init_weights:
            self.apply(weights_init())

    def forward(self, x):
        # Remember original spatial size for the final upscale
        orig_h, orig_w = x.shape[2], x.shape[3]

        # Bilinear downscale (e.g. 1024 → 512)
        x = F.interpolate(x, scale_factor=0.5, mode="bilinear", align_corners=True)

        # Encoder
        x1 = self.inc(x)        # 64ch  @ 512
        x2 = self.down1(x1)     # 128ch @ 256
        x3 = self.down2(x2)     # 256ch @ 128
        x4 = self.down3(x3)     # 256ch @ 64

        # Decoder
        x33 = self.up1(x4, x3)  # 128ch @ 128
        x22 = self.up2(x33, x2) # 64ch  @ 256
        x11 = self.up3(x22, x1) # 64ch  @ 512

        x0 = self.outc(x11)     # 3ch   @ 512
        x0 = self.last_layer_activation(x0)

        # Bilinear upscale back to original resolution (e.g. 512 → 1024)
        x0 = F.interpolate(x0, size=(orig_h, orig_w), mode="bilinear", align_corners=True)

        return x0


if __name__ == "__main__":
    print("=" * 60)
    print("BaseUNetHalf (current)")
    print("=" * 60)
    model = BaseUNetHalf(n_channels=3, n_classes=3)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters : {num_params:,}  ({num_params * 4 / 1024 / 1024:.1f} MB)")
    x = torch.randn(1, 3, 1024, 1024)
    out = model(x)
    print(f"  Input      : {x.shape}")
    print(f"  Output     : {out.shape}")

    print()
    print("=" * 60)
    print("BaseUNetHalfLite (new — bilinear down/up, one fewer level)")
    print("=" * 60)
    model_lite = BaseUNetHalfLite(n_channels=3, n_classes=3)
    num_params_lite = sum(p.numel() for p in model_lite.parameters())
    print(f"  Parameters : {num_params_lite:,}  ({num_params_lite * 4 / 1024 / 1024:.1f} MB)")
    out_lite = model_lite(x)
    print(f"  Input      : {x.shape}")
    print(f"  Output     : {out_lite.shape}")

    print()
    reduction = (1 - num_params_lite / num_params) * 100
    print(f"Parameter reduction: {reduction:.1f}%")