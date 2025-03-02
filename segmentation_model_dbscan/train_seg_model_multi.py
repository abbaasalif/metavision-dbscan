import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, models
import numpy as np
import matplotlib.pyplot as plt
import random
from catalyst import dl
from catalyst.contrib.nn.criterion.dice import DiceLoss
from catalyst.contrib.nn.criterion.trevsky import TrevskyLoss
import torchvision.transforms.functional as TF
from pathlib import Path
from typing import Any, Callable, Optional
from PIL import Image
from torchvision.datasets.vision import VisionDataset
from torchvision.models.segmentation.deeplabv3 import DeepLabHead

# ---------------------------
# Helper functions for metrics (now for 2 classes)
# ---------------------------
def compute_iou(pred, target, num_classes=2, eps=1e-7):
    """Compute per-class IoU."""
    ious = []
    for cls in range(num_classes):
        pred_inds = (pred == cls)
        target_inds = (target == cls)
        intersection = (pred_inds & target_inds).sum()
        union = (pred_inds | target_inds).sum()
        ious.append(intersection / (union + eps))
    return np.array(ious)

def compute_dice(pred, target, num_classes=2, eps=1e-7):
    """Compute per-class Dice coefficient."""
    dices = []
    for cls in range(num_classes):
        pred_inds = (pred == cls)
        target_inds = (target == cls)
        intersection = (pred_inds & target_inds).sum()
        dices.append((2 * intersection) / (pred_inds.sum() + target_inds.sum() + eps))
    return np.array(dices)

# ---------------------------
# 1. Helper: Convert RGB Mask to Integer Labels (remove red class)
# ---------------------------
def rgb_to_label_mask(mask):
    """
    Converts an RGB mask (PIL Image) with the following mapping:
      - Black   (0, 0, 0)        -> 0 (background)
      - White   (255, 255, 255)  -> 1 (e.g. rocket)
      
    Any red pixels (255, 0, 0) are now treated as background.
    """
    mask_np = np.array(mask)
    # Start with all zeros (background)
    label_mask = np.zeros(mask_np.shape[:2], dtype=np.uint8)
    # Map white pixels to label 1
    label_mask[np.all(mask_np == [255, 255, 255], axis=-1)] = 1
    # Red pixels will remain 0 (background)
    return label_mask

# ---------------------------
# 2. Custom Transform for Segmentation (for training)
# ---------------------------
class CustomTransform(object):
    """
    Applies resizing, random cropping, and flipping to both image and mask.
    For the mask, instead of converting to tensor directly (which scales to [0,1]),
    we convert it into an integer label mask.
    """
    def __init__(self, output_size, crop_size):
        assert isinstance(output_size, (int, tuple))
        assert isinstance(crop_size, (int, tuple))
        self.output_size = output_size
        self.crop_size = crop_size
        self.resize = transforms.Resize(self.output_size)

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        # Resize both image and mask
        image = self.resize(image)
        mask = self.resize(mask)
        # Random crop parameters
        i, j, h, w = transforms.RandomCrop.get_params(image, output_size=self.crop_size)
        image = TF.crop(image, i, j, h, w)
        mask = TF.crop(mask, i, j, h, w)
        # Random horizontal flip
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        # Random vertical flip
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)
        # Convert image to tensor (scales to [0,1])
        image = TF.to_tensor(image)
        # Convert mask to integer labels (do NOT use TF.to_tensor here)
        mask = rgb_to_label_mask(mask)
        mask = torch.from_numpy(mask).long()
        return {'image': image, 'mask': mask}

# ---------------------------
# 3. Dataset Definition
# ---------------------------
class SegmentationDataset(VisionDataset):
    """
    A dataset for segmentation tasks. It loads images and masks.
    The masks are assumed to be in RGB and will be converted to integer labels.
    """
    def __init__(self,
                 root: str,
                 image_folder: str,
                 mask_folder: str,
                 transforms: Optional[Callable] = None,
                 seed: int = None,
                 fraction: float = None,
                 subset: str = None,
                 train_set: bool = False,
                 image_color_mode: str = "rgb",
                 mask_color_mode: str = "rgb") -> None:
        super().__init__(root, transforms)
        image_folder_path = Path(self.root) / image_folder
        mask_folder_path = Path(self.root) / mask_folder
        if not image_folder_path.exists():
            raise OSError(f"{image_folder_path} does not exist.")
        if not mask_folder_path.exists():
            raise OSError(f"{mask_folder_path} does not exist.")
        if image_color_mode not in ["rgb", "grayscale"]:
            raise ValueError(f"{image_color_mode} is an invalid choice.")
        if mask_color_mode not in ["rgb", "grayscale"]:
            raise ValueError(f"{mask_color_mode} is an invalid choice.")
        self.image_color_mode = image_color_mode
        self.mask_color_mode = mask_color_mode
        if not fraction:
            self.image_names = sorted(image_folder_path.glob("*"))
            self.mask_names = sorted(mask_folder_path.glob("*"))
        else:
            if subset not in ["Train", "Test"]:
                raise ValueError(f"{subset} is not a valid input.")
            self.fraction = fraction
            self.image_list = np.array(sorted(image_folder_path.glob("*")))
            self.mask_list = np.array(sorted(mask_folder_path.glob("*")))
            if seed:
                np.random.seed(seed)
                indices = np.arange(len(self.image_list))
                np.random.shuffle(indices)
                self.image_list = self.image_list[indices]
                self.mask_list = self.mask_list[indices]
            if subset == "Train":
                self.image_names = self.image_list[:int(np.ceil(len(self.image_list) * (1 - self.fraction)))]
                self.mask_names = self.mask_list[:int(np.ceil(len(self.mask_list) * (1 - self.fraction)))]
            else:
                self.image_names = self.image_list[int(np.ceil(len(self.image_list) * (1 - self.fraction))):]
                self.mask_names = self.mask_list[int(np.ceil(len(self.mask_list) * (1 - self.fraction))):]
        self.train_set = train_set
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, index: int) -> Any:
        image_path = self.image_names[index]
        mask_path = self.mask_names[index]
        with open(image_path, "rb") as image_file, open(mask_path, "rb") as mask_file:
            image = Image.open(image_file)
            if self.image_color_mode == "rgb":
                image = image.convert("RGB")
            else:
                image = image.convert("L")
            mask = Image.open(mask_file)
            if self.mask_color_mode == "rgb":
                mask = mask.convert("RGB")
            else:
                mask = mask.convert("L")
            sample = {"image": image, "mask": mask}
            if self.transforms and self.train_set:
                sample = self.transforms(sample)
            elif self.transforms:
                # For validation, apply the transform to the image.
                sample["image"] = self.transforms(sample["image"])
                # Also resize the mask to ensure it matches the image dimensions.
                mask_resized = transforms.Resize((512, 512))(sample["mask"])
                sample["mask"] = rgb_to_label_mask(mask_resized)
                sample["mask"] = torch.from_numpy(sample["mask"]).long()
            return sample

# ---------------------------
# 4. Custom Callback to Compute Metrics and Log to TensorBoard
# ---------------------------
# (Callback is commented out, but can be adapted if needed)
# class SegmentationMetricsCallback(dl.Callback):
#     def __init__(self, order: int = 90):
#         super().__init__(order=order)
#     def on_loader_end(self, runner):
#         if runner.loader_key == "valid":
#             predictions = []
#             targets = []
#             for batch in runner.loader:
#                 x = batch["image"].to(runner.device)
#                 y_true = batch["mask"].to(runner.device)  # integer target: [B, H, W]
#                 with torch.no_grad():
#                     logits = runner.model(x)
#                     pred = torch.argmax(logits, dim=1)  # [B, H, W]
#                 predictions.append(pred.cpu().numpy())
#                 targets.append(y_true.cpu().numpy())
#             predictions = np.concatenate(predictions, axis=0)
#             targets = np.concatenate(targets, axis=0)
#             iou = compute_iou(predictions, targets, num_classes=2)
#             dice = compute_dice(predictions, targets, num_classes=2)
#             # Use runner.engine.logger to log metrics to TensorBoard
#             # runner.engine.logger.log_metrics({
#             #     "valid_iou_mean": float(np.nanmean(iou)),
#             #     "valid_dice_mean": float(np.nanmean(dice))
#             # }, runner.epoch)

# ---------------------------
# 5. Main: Define Transforms, Datasets, Model, and Train
# ---------------------------
if __name__ == "__main__":
    # Ensure Windows works correctly with multiple workers.
    import torch.multiprocessing
    torch.multiprocessing.freeze_support()

    # Define training transforms: joint transform for both image and mask.
    train_transforms = transforms.Compose([
        CustomTransform(output_size=(520, 520), crop_size=(512, 512))
    ])
    # For validation, we use basic transforms for the image.
    val_transforms = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])

    train_dataset = SegmentationDataset('./segmentation_new', 'images', 'masks',
                                        transforms=train_transforms,
                                        image_color_mode='rgb',
                                        mask_color_mode='rgb',
                                        train_set=True)
    val_dataset = SegmentationDataset('./segmentation_new_test', 'images', 'masks',
                                      transforms=val_transforms,
                                      image_color_mode='rgb',
                                      mask_color_mode='rgb')

    train_dataloader = DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=8, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=10, shuffle=False, num_workers=8)

    # ---------------------------
    # Model Definition (Remove Sigmoid for Multi-Class)
    # ---------------------------
    class SegModel(nn.Module):
        def __init__(self, backbone='resnet'):
            super().__init__()
            if backbone == 'resnet':
                deeplabv3_model = models.segmentation.deeplabv3_resnet50(weights='DEFAULT')
                # Change output to 2 classes (background and rocket)
                self.head = DeepLabHead(2048, 2)
            else:
                deeplabv3_model = models.segmentation.deeplabv3_mobilenet_v3_large(weights='DEFAULT')
                self.head = DeepLabHead(960, 2)
            self.backbone = deeplabv3_model.backbone
            for param in self.backbone.parameters():
                param.requires_grad = False
            # Do not use Sigmoid: we output raw logits.
        def forward(self, x):
            input_shape = x.shape[-2:]
            x = self.backbone(x)['out']
            x = self.head(x)
            x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
            return x  # raw logits [B, 2, H, W]

    model = SegModel(backbone='resnet')

    # ---------------------------
    # Training Setup with nn.CrossEntropyLoss
    # ---------------------------
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    model.to(device)

    # For nn.CrossEntropyLoss, the model output should be [B, C, H, W] and the target [B, H, W]
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)
    loaders = {'train': train_dataloader, 'valid': val_dataloader}

    # Custom Catalyst runner: we pass the integer target directly.
    class CustomRunner(dl.SupervisedRunner):
        def handle_batch(self, batch):
            x = batch['image'].to(device)
            y_pred = self.model(x)  # raw logits [B, 2, H, W]
            target = batch['mask'].to(device)  # shape [B, H, W] with integer labels (0 or 1)
            self.batch = {self._input_key: x, self._output_key: y_pred, self._target_key: target}

    runner = CustomRunner(input_key='features', output_key='scores', target_key='target', loss_key='loss')
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        loaders=loaders,
        num_epochs=100,
        callbacks=[
            dl.EarlyStoppingCallback(patience=3, loader_key='valid', metric_key='loss', minimize=True),
        ],
        logdir='./segementation_deeplab_resnet50_ce_loss',
        valid_loader='valid',
        valid_metric='loss',
        minimize_valid_metric=True,
        verbose=True,
        load_best_on_end=True
    )
    torch.save(model.state_dict(), './segmentation_new_model_multi/segmodel_resnet_ce_loss.pth')
