import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
import random
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ---------------------------
# Helper Functions: IoU and Dice (for 2 classes)
# ---------------------------
def compute_iou(pred, target, num_classes=2, eps=1e-7):
    """Compute per-class IoU for 2 classes."""
    ious = []
    for cls in range(num_classes):
        pred_inds = (pred == cls)
        target_inds = (target == cls)
        intersection = (pred_inds & target_inds).sum()
        union = (pred_inds | target_inds).sum()
        ious.append(intersection / (union + eps))
    return np.array(ious)

def compute_dice(pred, target, num_classes=2, eps=1e-7):
    """Compute per-class Dice coefficient for 2 classes."""
    dices = []
    for cls in range(num_classes):
        pred_inds = (pred == cls)
        target_inds = (target == cls)
        intersection = (pred_inds & target_inds).sum()
        dices.append((2 * intersection) / (pred_inds.sum() + target_inds.sum() + eps))
    return np.array(dices)

# ---------------------------
# Helper: Convert RGB Mask to Integer Labels (red is now background)
# ---------------------------
def rgb_to_label_mask(mask):
    """
    Converts an RGB mask (PIL Image) with the following mapping:
      - Black   (0, 0, 0)        -> 0 (background)
      - White   (255, 255, 255)  -> 1 (e.g. rocket)
      
    Any red pixels (255, 0, 0) are treated as background.
    """
    mask_np = np.array(mask)
    # Initialize with 0's for background
    label_mask = np.zeros(mask_np.shape[:2], dtype=np.uint8)
    # Map white pixels to label 1.
    label_mask[np.all(mask_np == [255, 255, 255], axis=-1)] = 1
    return label_mask

# ---------------------------
# Dataset Definition (Validation)
# ---------------------------
from torchvision.datasets.vision import VisionDataset
class SegmentationDataset(VisionDataset):
    """
    A dataset for segmentation tasks. It loads images and masks.
    For validation, it applies transforms to the image and resizes the mask.
    """
    def __init__(self,
                 root: str,
                 image_folder: str,
                 mask_folder: str,
                 transforms=None,
                 image_color_mode: str = "rgb",
                 mask_color_mode: str = "rgb") -> None:
        super().__init__(root, transforms)
        image_folder_path = Path(self.root) / image_folder
        mask_folder_path = Path(self.root) / mask_folder
        if not image_folder_path.exists():
            raise OSError(f"{image_folder_path} does not exist.")
        if not mask_folder_path.exists():
            raise OSError(f"{mask_folder_path} does not exist.")
        self.image_color_mode = image_color_mode
        self.mask_color_mode = mask_color_mode
        self.image_names = sorted(list(image_folder_path.glob("*")))
        self.mask_names = sorted(list(mask_folder_path.glob("*")))
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, index: int):
        image_path = self.image_names[index]
        mask_path = self.mask_names[index]
        with open(image_path, "rb") as image_file, open(mask_path, "rb") as mask_file:
            image = Image.open(image_file)
            image = image.convert("RGB") if self.image_color_mode == "rgb" else image.convert("L")
            mask = Image.open(mask_file)
            mask = mask.convert("RGB") if self.mask_color_mode == "rgb" else mask.convert("L")
        sample = {"image": image, "mask": mask}
        if self.transforms:
            # Apply transforms to the image
            sample["image"] = self.transforms(sample["image"])
            # Resize mask to ensure it matches image dimensions and convert to integer labels
            mask_resized = transforms.Resize((512, 512))(sample["mask"])
            sample["mask"] = rgb_to_label_mask(mask_resized)
            sample["mask"] = torch.from_numpy(sample["mask"]).long()
        return sample

# ---------------------------
# Model Definition (updated for 2 classes)
# ---------------------------
class SegModel(nn.Module):
    def __init__(self, backbone='resnet'):
        super().__init__()
        if backbone == 'resnet':
            deeplabv3_model = models.segmentation.deeplabv3_resnet50(weights='DEFAULT')
            # Update head to output 2 channels: background and rocket.
            self.head = DeepLabHead(2048, 2)
        else:
            deeplabv3_model = models.segmentation.deeplabv3_mobilenet_v3_large(weights='DEFAULT')
            self.head = DeepLabHead(960, 2)
        self.backbone = deeplabv3_model.backbone
        # Freeze backbone parameters.
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x):
        input_shape = x.shape[-2:]
        x = self.backbone(x)['out']
        x = self.head(x)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        return x  # raw logits [B, 2, H, W]

# ---------------------------
# Load Saved Model
# ---------------------------
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = SegModel(backbone='resnet')
model_path = './segmentation_new_model_multi/segmodel_resnet_ce_loss.pth'
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# ---------------------------
# Prepare the Validation Dataset and DataLoader
# ---------------------------
val_transforms = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])

val_dataset = SegmentationDataset(root='./segmentation_new_test',
                                  image_folder='images',
                                  mask_folder='masks',
                                  transforms=val_transforms,
                                  image_color_mode='rgb',
                                  mask_color_mode='rgb')

# Updated colormap for 2 classes: background (black) and rocket (white)
cmap = mcolors.ListedColormap([(0, 0, 0), (1, 1, 1)])

# ---------------------------
# Warm-up the Model (important for accurate timing, especially on GPU)
# ---------------------------
if device.type == 'cuda':
    dummy_input = torch.randn(1, 3, 512, 512).to(device)
    for _ in range(5):
        with torch.no_grad():
            _ = model(dummy_input)
    torch.cuda.synchronize()

# ---------------------------
# Evaluate Metrics and Inference Time over the Entire Validation Set
# ---------------------------
all_ious = []
all_dices = []
inference_times = []  # list to store inference time per sample

for idx in range(len(val_dataset)):
    sample = val_dataset[idx]
    image_tensor = sample['image']
    gt_mask = sample['mask']
    
    image = image_tensor.unsqueeze(0).to(device)
    
    # Timing the inference for each sample.
    if device.type == 'cuda':
        torch.cuda.synchronize()  # ensure previous GPU work is done
    start_time = time.time()
    with torch.no_grad():
        logits = model(image)
    if device.type == 'cuda':
        torch.cuda.synchronize()  # wait for GPU to finish computation
    end_time = time.time()
    
    inference_times.append(end_time - start_time)
    
    pred_mask = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
    gt_mask_np = gt_mask.cpu().numpy()
    
    iou = compute_iou(pred_mask, gt_mask_np, num_classes=2)
    dice = compute_dice(pred_mask, gt_mask_np, num_classes=2)
    
    all_ious.append(iou)
    all_dices.append(dice)

all_ious = np.array(all_ious)
all_dices = np.array(all_dices)

mean_iou_per_class = np.mean(all_ious, axis=0)
mean_dice_per_class = np.mean(all_dices, axis=0)
overall_mean_iou = np.mean(mean_iou_per_class)
overall_mean_dice = np.mean(mean_dice_per_class)

print("Final Metrics over the entire validation set:")
print(f"IoU per class: {mean_iou_per_class}")
print(f"Overall Mean IoU: {overall_mean_iou:.4f}")
print(f"Dice per class: {mean_dice_per_class}")
print(f"Overall Mean Dice: {overall_mean_dice:.4f}")

# Calculate and print the average inference time per sample.
avg_inference_time = np.mean(inference_times)
print(f"Average Inference Time per sample: {avg_inference_time:.4f} seconds")

# ---------------------------
# Save 10 Random Example Images with Inference Results
# ---------------------------
save_dir = "saved_examples"
os.makedirs(save_dir, exist_ok=True)

# Select 10 random indices
example_indices = np.random.choice(len(val_dataset), 10, replace=False)

for idx in example_indices:
    sample = val_dataset[idx]
    image_tensor = sample['image']
    gt_mask = sample['mask']
    
    # Convert image tensor to numpy for display.
    image_np = image_tensor.cpu().permute(1, 2, 0).numpy()
    
    image = image_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(image)
    pred_mask = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
    gt_mask_np = gt_mask.cpu().numpy()
    
    # Create a figure showing input, ground truth, and predicted mask.
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(image_np)
    plt.title("Input Image")
    plt.axis("off")
    
    plt.subplot(1, 3, 2)
    plt.imshow(gt_mask_np, cmap=cmap, vmin=0, vmax=1)
    plt.title("Ground Truth Mask")
    plt.axis("off")
    
    plt.subplot(1, 3, 3)
    plt.imshow(pred_mask, cmap=cmap, vmin=0, vmax=1)
    plt.title("Predicted Mask")
    plt.axis("off")
    
    # Save the figure to the designated directory.
    save_path = os.path.join(save_dir, f"example_{idx}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved example image: {save_path}")
