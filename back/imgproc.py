import math
import random
from typing import Any

import cv2
import numpy as np
import torch
from numpy import ndarray
from torch import Tensor

__all__ = [
    "image_to_tensor", "tensor_to_image",
     "preprocess_one_image",
     ]


def image_to_tensor(image: ndarray, range_norm: bool, half: bool) -> Tensor:
    """Convert the image data type to the Tensor (NCWH) data type supported by PyTorch

    Args:
        image (np.ndarray): The image data read by ``OpenCV.imread``, the data range is [0,255] or [0, 1]
        range_norm (bool): Scale [0, 1] data to between [-1, 1]
        half (bool): Whether to convert torch.float32 similarly to torch.half type

    Returns:
        tensor (Tensor): Data types supported by PyTorch

    Examples:
        >>> example_image = cv2.imread("lr_image.bmp")
        >>> example_tensor = image_to_tensor(example_image, range_norm=True, half=False)

    """
    # Convert image data type to Tensor data type
    tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float()

    # Scale the image data from [0, 1] to [-1, 1]
    if range_norm:
        tensor = tensor.mul(2.0).sub(1.0)

    # Convert torch.float32 image data type to torch.half image data type
    if half:
        tensor = tensor.half()

    return tensor


def tensor_to_image(tensor: Tensor, range_norm: bool, half: bool) -> Any:
    """将PyTorch支持的Tensor(NCWH)数据类型转换为np.ndarray(WHC)图像数据类型

    Args:
        tensor (Tensor): 由PyTorch支持的数据类型（NCHW），数据范围为[0, 1]
        range_norm (bool): 将[-1, 1]数据缩放到[0, 1]之间
        half (bool): 是否将torch.float32类似地转换为torch.half类型。

    Returns:
        image (np.ndarray): 由PIL或OpenCV支持的数据类型

    示例：
        >>> example_image = cv2.imread("lr_image.bmp")
        >>> example_tensor = image_to_tensor(example_image, range_norm=False, half=False)

    """
    if range_norm:
        tensor = tensor.add(1.0).div(2.0)
    if half:
        tensor = tensor.half()

    image = tensor.squeeze(0).permute(1, 2, 0).mul(255).clamp(0, 255).cpu().numpy().astype("uint8")

    return image


def preprocess_one_image(
        image_path: str,
        range_norm: bool,
        half: bool,
        device: torch.device,
) -> Tensor:
    """预处理图像数据

    Args:
        image_path (str): 图像的路径
        range_norm (bool): 将[0, 1]数据缩放到[-1, 1]之间
        half (bool): 是否将torch.float32类似地转换为torch.half类型
        device (torch.device): 模型所在的设备

    Returns: 图像的Tensor

    """
    if isinstance(image_path, str):
        image = cv2.imread(image_path).astype(np.float32) / 255.0
    else:
        image = cv2.imdecode(np.frombuffer(image_path.read(), np.uint8), cv2.IMREAD_COLOR)
        image = image.astype(np.float32) / 255.0

    # BGR转RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 将图像数据转换为pytorch格式数据
    tensor = image_to_tensor(image, range_norm, half).unsqueeze_(0)

    # 将tensor通道图像格式数据传输到CUDA设备
    tensor = tensor.to(device, non_blocking=True)

    return tensor
