import os
import cv2
import torch
from torch import nn
import imgproc
import model
from utils import load_state_dict
def generate_super_resolution_image(input_path, lr_output_path, sr_output_path: str, model_arch_name: str = "srresnet_x4", model_weights_path: str = "./weight/SRGAN_x4-ImageNet-8c4a7569.pth.tar", device_type: str = "cuda") -> None:


    device = torch.device(device_type)

    # Initialize model
    sr_model = model.__dict__[model_arch_name](
        in_channels=3, out_channels=3, channels=64, num_rcb=16
    )
    sr_model = sr_model.to(device=device)

    # Load model weights
    sr_model = load_state_dict(sr_model, model_weights_path)
    print(
        f"Load `{model_arch_name}` model weights `{os.path.abspath(model_weights_path)}` successfully."
    )

    # Enable evaluation mode
    sr_model.eval()

    if isinstance(input_path, str):
        lr_image = cv2.imread(input_path)
    else:
        lr_image = input_path


    lr_tensor = imgproc.preprocess_one_image(
        input_path, False, False, device=device
    )    
    lr_image = imgproc.tensor_to_image(lr_tensor, False, False)
    lr_image = cv2.cvtColor(lr_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(lr_output_path, lr_image)

    # Use model to generate super-resolution image
    with torch.no_grad():
        sr_tensor = sr_model(lr_tensor)

    # Save image

    sr_image = imgproc.tensor_to_image(sr_tensor, False, False)
    sr_image = cv2.cvtColor(sr_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(sr_output_path, sr_image)

    print(f"SR image saved to `{sr_output_path}`")


"""
lr_image = cv2.imread("./image/lr/flower_lr.png")
generate_super_resolution_image(
    input_path=lr_image,
    output_path="./image/sr/flower_sr.png",
    model_arch_name="srresnet_x4",
    model_weights_path="./weight/SRGAN_x4-ImageNet-8c4a7569.pth.tar",
    device_type="cuda"
)
"""
