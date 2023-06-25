import math
from typing import Any

import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F_torch
from torchvision import models
from torchvision import transforms
from torchvision.models.feature_extraction import create_feature_extractor

__all__ = [
    "SRResNet",
    "Discriminator",
    "srresnet_x4",
    "discriminator",
    "content_loss",
]


# 超分辨率残差网络
class SRResNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        channels: int,
        num_rcb: int,
        upscale_factor: int,
    ) -> None:
        super(SRResNet, self).__init__()
        # 低频信息提取层
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, channels, (9, 9), (1, 1), (4, 4)),
            nn.PReLU(),
        )

        # 高频信息提取块
        trunk = []
        for _ in range(num_rcb):
            trunk.append(_ResidualConvBlock(channels))
        self.trunk = nn.Sequential(*trunk)

        # 高频信息线性融合层
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(channels),
        )

        # 缩放块
        upsampling = []
        if upscale_factor == 2 or upscale_factor == 4 or upscale_factor == 8:
            for _ in range(int(math.log(upscale_factor, 2))):
                upsampling.append(_UpsampleBlock(channels, 2))
        elif upscale_factor == 3:
            upsampling.append(_UpsampleBlock(channels, 3))
        self.upsampling = nn.Sequential(*upsampling)

        # 重建块
        self.conv3 = nn.Conv2d(channels, out_channels, (9, 9), (1, 1), (4, 4))

        # 初始化神经网络权重
        self._initialize_weights()

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

    # 支持torch.script函数
    def _forward_impl(self, x: Tensor) -> Tensor:
        out1 = self.conv1(x)
        out = self.trunk(out1)
        out2 = self.conv2(out)
        out = torch.add(out1, out2)
        out = self.upsampling(out)
        out = self.conv3(out)

        out = torch.clamp_(out, 0.0, 1.0)

        return out

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)


# 判别器
class Discriminator(nn.Module):
    def __init__(self) -> None:
        super(Discriminator, self).__init__()
        self.features = nn.Sequential(
            # 输入尺寸. (3) x 96 x 96
            nn.Conv2d(3, 64, (3, 3), (1, 1), (1, 1), bias=True),
            nn.LeakyReLU(0.2, True),
            # 状态尺寸. (64) x 48 x 48
            nn.Conv2d(64, 64, (3, 3), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            # 状态尺寸. (128) x 24 x 24
            nn.Conv2d(128, 128, (3, 3), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            # 状态尺寸. (256) x 12 x 12
            nn.Conv2d(256, 256, (3, 3), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 512, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            # 状态尺寸. (512) x 6 x 6
            nn.Conv2d(512, 512, (3, 3), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 * 6 * 6, 1024),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1024, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        # 输入图像尺寸必须等于96
        assert x.shape[2] == 96 and x.shape[3] == 96, "Image shape must equal 96x96"

        out = self.features(x)
        out = torch.flatten(out, 1)
        out = self.classifier(out)

        return out


# 残差卷积块
class _ResidualConvBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super(_ResidualConvBlock, self).__init__()
        self.rcb = nn.Sequential(
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.rcb(x)

        out = torch.add(out, identity)

        return out


# 上采样块
class _UpsampleBlock(nn.Module):
    def __init__(self, channels: int, upscale_factor: int) -> None:
        super(_UpsampleBlock, self).__init__()
        self.upsample_block = nn.Sequential(
            nn.Conv2d(
                channels,
                channels * upscale_factor * upscale_factor,
                (3, 3),
                (1, 1),
                (1, 1),
            ),
            nn.PixelShuffle(2),
            nn.PReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        out = self.upsample_block(x)

        return out


# 内容损失
class _ContentLoss(nn.Module):
    """基于VGG19网络构建内容损失函数。
    使用后面层次的高级特征映射层将更加关注图像的纹理内容。
    """

    def __init__(
        self,
        feature_model_extractor_node: str,
        feature_model_normalize_mean: list,
        feature_model_normalize_std: list,
    ) -> None:
        super(_ContentLoss, self).__init__()
        # 获取指定特征提取节点的名称
        self.feature_model_extractor_node = feature_model_extractor_node
        # 加载在ImageNet数据集上训练的VGG19模型。
        model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        # 提取VGG19模型中的第三十六层输出作为内容损失。
        self.feature_extractor = create_feature_extractor(
            model, [feature_model_extractor_node]
        )
        # 设置为验证模式
        self.feature_extractor.eval()

        # 输入数据的预处理方法。
        # 这是ImageNet数据集的VGG模型预处理方法。
        self.normalize = transforms.Normalize(
            feature_model_normalize_mean, feature_model_normalize_std
        )

        # 冻结模型参数。
        for model_parameters in self.feature_extractor.parameters():
            model_parameters.requires_grad = False

    def forward(self, sr_tensor: Tensor, gt_tensor: Tensor) -> Tensor:
        # 标准化操作
        sr_tensor = self.normalize(sr_tensor)
        gt_tensor = self.normalize(gt_tensor)

        sr_feature = self.feature_extractor(sr_tensor)[
            self.feature_model_extractor_node
        ]
        gt_feature = self.feature_extractor(gt_tensor)[
            self.feature_model_extractor_node
        ]

        # 找到两个图像之间的特征图差异
        loss = F_torch.mse_loss(sr_feature, gt_feature)

        return loss


# 超分辨率残差网络x4
def srresnet_x4(**kwargs: Any) -> SRResNet:
    model = SRResNet(upscale_factor=4, **kwargs)

    return model


# 判别器
def discriminator() -> Discriminator:
    model = Discriminator()

    return model


# 内容损失
def content_loss(**kwargs: Any) -> _ContentLoss:
    content_loss = _ContentLoss(**kwargs)

    return content_loss
