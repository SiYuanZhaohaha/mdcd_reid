import os
import torch
from torchvision.transforms.functional import to_tensor, to_pil_image

from .src.model import (ResidualDiffusion,Trainer, Unet, UnetRes,set_seed)

class ImageRestorationTransform:
    def __init__(self):
        """
        Initialize the transform with the restoration model.
        Args:
            model: The trained image restoration model (should have a `sample` method).
            device: The device on which the model will run (e.g., 'cuda' or 'cpu').
        """
        num_unet = 1
        objective = 'pred_res'
        test_res_or_noise = "res"
        sampling_timesteps = 3
        sum_scale = 0.01
        ddim_sampling_eta = 0.
        delta_end = 1.8e-3
        condition = True
        image_size = 256 #[256, 128]
        self.unet_mode = UnetRes(
			dim=32,
			dim_mults=(1, 1, 1, 1),
			num_unet=num_unet,
			condition=condition,
			objective=objective,
			test_res_or_noise = test_res_or_noise
		)
        self.model = ResidualDiffusion(
				self.unet_mode,
				image_size=image_size,
				timesteps=1000,           # number of steps
				delta_end = delta_end,
				sampling_timesteps=sampling_timesteps,
				ddim_sampling_eta=ddim_sampling_eta,
				objective=objective,
				loss_type='l1',            # L1 or L2
				condition=condition,
				sum_scale=sum_scale,
				test_res_or_noise = test_res_or_noise,
		)
        self.device = "cuda"
        ckpt_path = os.environ.get("DIFFUIR_CKPT", "./pretrained/diffuir/model-9.pt")
        data = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(data['model'])
        self.model.eval()  # Set the model to evaluation mode


    def __call__(self, image):
        """
        Apply image restoration on the input image.
        Args:
            image: PIL.Image or Tensor. The input image to be restored.
        Returns:
            PIL.Image: The restored image as a PIL.Image.
        """
        # Convert input image to Tensor if it's a PIL.Image
        if not isinstance(image, torch.Tensor):
            image = to_tensor(image).unsqueeze(0).to(self.device)  # Add batch dimension
        else:
            image = image.unsqueeze(0).to(self.device)  # Ensure batch dimension
		 # 确保模型和张量在同一设备上
        self.model = self.model.to(self.device)
        # Run the model to restore the image
        with torch.no_grad():
            restored_images = self.model.sample(image, batch_size=1, last=True)  # Call model's `sample` method

        # Extract the last image output
        restored_image = restored_images[-1].squeeze(0).cpu()  # Remove batch dimension

        # Convert back to PIL.Image
        return to_pil_image(restored_image)
    
import torch
import torch.nn.functional as F
from math import log10
from torchvision import transforms

def calculate_psnr(img1, img2, max_val=1.0):
    """
    Calculate PSNR between two images.
    Args:
        img1: torch.Tensor, [C, H, W] or [N, C, H, W]
        img2: torch.Tensor, same shape as img1
        max_val: Maximum possible pixel value (default 1.0 for normalized images).
    Returns:
        PSNR value.
    """
    mse = F.mse_loss(img1, img2)
    psnr = 10 * log10(max_val ** 2 / mse.item())
    return psnr


def calculate_ssim(img1, img2, max_val=1.0):
    """
    Calculate SSIM between two images using PyTorch.
    Args:
        img1: torch.Tensor, [C, H, W] or [N, C, H, W]
        img2: torch.Tensor, same shape as img1
        max_val: Maximum possible pixel value (default 1.0 for normalized images).
    Returns:
        SSIM value.
    """
    # Constants for SSIM calculation
    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2

    # Mean
    mu1 = F.avg_pool2d(img1, kernel_size=3, stride=1, padding=1)
    mu2 = F.avg_pool2d(img2, kernel_size=3, stride=1, padding=1)

    # Variance and Covariance
    sigma1_sq = F.avg_pool2d(img1 * img1, kernel_size=3, stride=1, padding=1) - mu1 ** 2
    sigma2_sq = F.avg_pool2d(img2 * img2, kernel_size=3, stride=1, padding=1) - mu2 ** 2
    sigma12 = F.avg_pool2d(img1 * img2, kernel_size=3, stride=1, padding=1) - mu1 * mu2

    # SSIM formula
    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean().item()

if __name__ == "__main__":
    from torchvision import transforms
    from PIL import Image
    restoration_transform = ImageRestorationTransform()
    """ image_transforms = transforms.Compose([
		restoration_transform,  # Apply the restoration model
		transforms.Resize((256, 256)),  # Resize after restoration
		transforms.ToTensor(),  # Convert to Tensor for further processing
		transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # Normalize image
	]) """
    image_path = os.environ.get("RESTORE_INPUT", "./examples/hazy_input.jpg")
    gt_path = os.environ.get("RESTORE_GT", "./examples/clean_gt.jpg")
    output_path = os.environ.get("RESTORE_OUTPUT", "restored.jpg")
    image = Image.open(image_path).convert("RGB")
    restored_image = restoration_transform(image)
    input_tensor = to_tensor(image)
    restored_tensor = to_tensor(restored_image)
    #diff_tensor = torch.abs(input_tensor - restored_tensor)
    gt = Image.open(gt_path).convert("RGB")
    gt = to_tensor(gt)
    restored_image.save(output_path)
    SSIM = calculate_ssim(gt, restored_tensor)
    psnr = calculate_psnr(gt, restored_tensor)
    print("SSIM", SSIM)
    print("psnr", psnr)
    print(f"Restored image save")
