# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.utils import make_grid
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
# %matplotlib inline
torch.backends.cudnn.benchmark=True
from catalyst import dl
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from catalyst.contrib.nn.criterion.iou import IoULoss
from catalyst.contrib.nn.criterion.dice import DiceLoss
from catalyst.contrib.nn.criterion.trevsky import TrevskyLoss
import torchvision.transforms.functional as TF
from pathlib import Path
from typing import Any, Callable, Optional
from PIL import Image
from torchvision.datasets.vision import VisionDataset
import random
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
# %%
class CustomTransform(object):
  """ Rescale the image in a sample to a given size.

  Args:
    output_size (tuple or int): Desired output size. If tuple, output is
    matched to output_size. If int, smaller of image edges is matched.

    crop_size (tuple or int): Desired size for random cropping the image.
  """
  def __init__(self, output_size, crop_size):
    assert isinstance(output_size, (int, tuple))
    assert isinstance(crop_size, (int, tuple))
    self.output_size = output_size
    self.crop_size = crop_size

  def __call__(self, sample):
    image, mask = sample['image'], sample['mask']
    #Resize
    resize = transforms.Resize(self.output_size)
    image = resize(image)
    mask = resize(mask)

    #Random crop
    i, j, h, w = transforms.RandomCrop.get_params(image, output_size=self.crop_size)
    image = TF.crop(image, i, j, h, w)
    mask = TF.crop(mask, i, j, h, w)

    #Random horizontal flipping
    if random.random() > 0.5:
      image = TF.hflip(image)
      mask = TF.hflip(mask)

    #Random vertical flipping
    if random.random() > 0.5:
      image = TF.vflip(image)
      mask = TF.vflip(mask)
    image = TF.to_tensor(image)
    mask = TF.to_tensor(mask)

    return {'image': image, 'mask': mask}
# %%
class SegModel(torch.nn.Module):
    """
    Adapted from
    https://github.com/msminhas93/DeepLabv3FineTuning/blob/64541451e85d61dea66/model.py

    And from looking at the source code for the deeplabv3_resnet50 module:

    import inspect
    from torchvision. import models

    model = models.segmentation.deeplabv3_resnet50(pretrained=True)

    print(inspect.getsource(model.backbone.forward))
    print(inspect.getsource(model.forward))

    """
    def __init__(self, backbone='resnet'):
        super().__init__()
        self.backbone = backbone
        if backbone == 'resnet':
            deeplabv3_model = models.segmentation.deeplabv3_resnet50(weights='DEFAULT')
            self.head = DeepLabHead(2048, 3)
        else:
            deeplabv3_model = models.segmentation.deeplabv3_mobilenet_v3_large(weights='DEFAULT')
            self.head = DeepLabHead(960, 3)

        self.backbone = deeplabv3_model.backbone
        self.backbone.trainable = False
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        # see 'inspect.getsource(deeplabv3_model.forward)'
        input_shape = x.shape[-2:]
        x = self.backbone(x)['out']
        x = self.head(x)
        x = torch.nn.functional.interpolate(
            x, size=input_shape, mode='bilinear', align_corners=False
        )
        return self.sigmoid(x)

class SegmentationDataset(VisionDataset):
    """A PyTorch dataset for image segmentation task.
    The dataset is compatible with torchvision transforms.
    The transforms passed would be applied to both the Images and Masks.
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
                 mask_color_mode: str = "grayscale") -> None:
        """
        Args:
            root (str): Root directory path.
            image_folder (str): Name of the folder that contains the images in the root directory.
            mask_folder (str): Name of the folder that contains the masks in the root directory.
            transforms (Optional[Callable], optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.ToTensor`` for images. Defaults to None.
            seed (int, optional): Specify a seed for the train and test split for reproducible results. Defaults to None.
            fraction (float, optional): A float value from 0 to 1 which specifies the validation split fraction. Defaults to None.
            subset (str, optional): 'Train' or 'Test' to select the appropriate set. Defaults to None.
            image_color_mode (str, optional): 'rgb' or 'grayscale'. Defaults to 'rgb'.
            mask_color_mode (str, optional): 'rgb' or 'grayscale'. Defaults to 'grayscale'.

        Raises:
            OSError: If image folder doesn't exist in root.
            OSError: If mask folder doesn't exist in root.
            ValueError: If subset is not either 'Train' or 'Test'
            ValueError: If image_color_mode and mask_color_mode are either 'rgb' or 'grayscale'
        """
        super().__init__(root, transforms)
        image_folder_path = Path(self.root) / image_folder
        mask_folder_path = Path(self.root) / mask_folder
        if not image_folder_path.exists():
            raise OSError(f"{image_folder_path} does not exist.")
        if not mask_folder_path.exists():
            raise OSError(f"{mask_folder_path} does not exist.")

        if image_color_mode not in ["rgb", "grayscale"]:
            raise ValueError(
                f"{image_color_mode} is an invalid choice. Please enter from rgb grayscale."
            )
        if mask_color_mode not in ["rgb", "grayscale"]:
            raise ValueError(
                f"{mask_color_mode} is an invalid choice. Please enter from rgb grayscale."
            )

        self.image_color_mode = image_color_mode
        self.mask_color_mode = mask_color_mode

        if not fraction:
            self.image_names = sorted(image_folder_path.glob("*"))
            self.mask_names = sorted(mask_folder_path.glob("*"))
        else:
            if subset not in ["Train", "Test"]:
                raise (ValueError(
                    f"{subset} is not a valid input. Acceptable values are Train and Test."
                ))
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
                self.image_names = self.image_list[:int(
                    np.ceil(len(self.image_list) * (1 - self.fraction)))]
                self.mask_names = self.mask_list[:int(
                    np.ceil(len(self.mask_list) * (1 - self.fraction)))]
            else:
                self.image_names = self.image_list[
                    int(np.ceil(len(self.image_list) * (1 - self.fraction))):]
                self.mask_names = self.mask_list[
                    int(np.ceil(len(self.mask_list) * (1 - self.fraction))):]
        self.train_set = train_set
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, index: int) -> Any:
        image_path = self.image_names[index]
        mask_path = self.mask_names[index]
        with open(image_path, "rb") as image_file, open(mask_path,
                                                        "rb") as mask_file:
            image = Image.open(image_file)
            if self.image_color_mode == "rgb":
                image = image.convert("RGB")
            elif self.image_color_mode == "grayscale":
                image = image.convert("L")
            mask = Image.open(mask_file)
            if self.mask_color_mode == "rgb":
                mask = mask.convert("RGB")
            elif self.mask_color_mode == "grayscale":
                mask = mask.convert("L")
            sample = {"image": image, "mask": mask}
            if self.transforms and self.train_set:
              sample = self.transforms(sample)
            elif self.transforms:
                sample["image"] = self.transforms(sample["image"])
                sample["mask"] = self.transforms(sample["mask"])
        return sample

train_transforms = transforms.Compose(
    [
       CustomTransform(output_size=(520, 520), crop_size=(512, 512))
    ]
)

val_transforms = transforms.Compose(
    [
        transforms.Resize((512, 512)), transforms.ToTensor(),
    ]
)


if __name__ == '__main__':
    train_dataset = SegmentationDataset('./segmentation_new', 'images','masks', transforms=train_transforms, image_color_mode='rgb', mask_color_mode='rgb', train_set=True)
    val_dataset = SegmentationDataset('./segmentation_new_test', 'images', 'masks', transforms=val_transforms, image_color_mode='rgb', mask_color_mode='rgb')

    train_dataloader = DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=8, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=10, shuffle=False, num_workers=8)




    # %%
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

    
    model = SegModel(backbone='resnet')

    criterion = IoULoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)
    loaders = {'train': train_dataloader, 'valid':val_dataloader}

    class CustomRunner(dl.SupervisedRunner):
        def handle_batch(self, batch):
            x = batch['image']
            y_pred = self.model(x)
            target = batch['mask']
            self.batch = {self._input_key:x, self._output_key: y_pred, self._target_key: target}

    runner = CustomRunner(input_key='features', output_key='scores', target_key='target', loss_key='loss')
    #model training
    runner.train(
        model=model,
        criterion = criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        loaders=loaders,
        num_epochs=100,
        callbacks=[
            dl.IOUCallback(input_key='scores', target_key='target'),
            dl.DiceCallback(input_key='scores', target_key='target'),
            dl.TrevskyCallback(input_key='scores', target_key='target', alpha=0.2),
            dl.EarlyStoppingCallback(patience=3, loader_key='valid', metric_key='loss', minimize=True),
        ],
        logdir='./segementation_deeplab_resnet50_iou',
        valid_loader='valid',
        valid_metric='loss',
        minimize_valid_metric=True,
        verbose=True,
        load_best_on_end=True
    )
    torch.save(model.state_dict(), './segmentation_new_model/segmodel_resnet_iou.pth')