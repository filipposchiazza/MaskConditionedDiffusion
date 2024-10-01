import os
import torch
import random
import torch.utils.data as data
from torchvision.io import read_image
from torchvision.datasets import ImageFolder


# Dataset for training the Diffusion model conditioned on the cell masks
class ImageWithMaskDataset(data.Dataset):

    def __init__(self, img_dir, transform=None, fraction=1.0, normalization_mode=1):
        """Image dataset for training and validation.
        Remark: Images are expected to have the masks stacked in the 2nd channel (side by side).
        
        Parameters
        ----------
        img_dir : str
            Path to the folder containing the images.
        transform : torchvision.transforms
            Image transformation.
        fraction : float
            Fraction of the dataset to use.
        normalization_mode : int
            1: Normalize the image to [0, 1]
            2: Normalize the image to [-1, 1]
        """
        self.img_dir = img_dir
        self.transform = transform
        self.fraction = fraction
        if normalization_mode not in [1, 2]:
            raise ValueError('Invalid normalization mode')
        else:
            self.normalization_mode = normalization_mode
        self.img_filenames = os.listdir(img_dir)
        self.img_filenames = self.img_filenames[:int(len(self.img_filenames) * self.fraction)]

    def __len__(self):
        return len(self.img_filenames)
        
    def __getitem__(self, idx):
        img_name = self.img_filenames[idx]
        img_path = os.path.join(self.img_dir, img_name)
        full_img = read_image(img_path)
        full_img = full_img[:3, :, :]
        img = full_img[:, :, :full_img.shape[2] // 2]
        mask = full_img[0, :, full_img.shape[2] // 2:] / 255.0
        if self.normalization_mode == 1:
            img = img / 255.0
        elif self.normalization_mode == 2:
            img = (img / 127.5) - 1
        return img, mask.unsqueeze(0)
        
    

def prepare_ImageDataset(img_dir, 
                         batch_size,
                         validation_split,
                         transform=None,
                         seed=123, 
                         fraction=1.0,
                         normalization_mode=1):
    """Prepare the image dataset for training and validation.

    Parameters
    ----------
    img_dir : str
        Path to the folder containing the images.
    batch_size : int
        Batch size.
    validation_split : float
        Fraction of the dataset to use for validation.
    transform : torchvision.transforms
        Image transformation.
    seed : int
        Random seed.
    fraction : float
        Fraction of the dataset to use.
    normalization_mode : int
        1: Normalize the image to [0, 1]
        2: Normalize the image to [-1, 1]

    Returns
    -------
    train_dataset : ImageWithMaskDataset
        Training dataset object.
    val_dataset : ImageWithMaskDataset
        Validation dataset object.
    train_dataloader : torch.utils.data.DataLoader
        Training DataLoader object.
    val_dataloader : torch.utils.data.DataLoader
        Validation DataLoader object.
    """
    dataset = ImageWithMaskDataset(img_dir, transform, fraction, normalization_mode)
    val_len = int(len(dataset) * validation_split)
    train_len = len(dataset) - val_len
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = data.random_split(dataset, 
                                                   lengths=[train_len, val_len], 
                                                   generator=generator)
    train_dataloader = data.DataLoader(train_dataset, 
                                       batch_size=batch_size, 
                                       shuffle=True, 
                                       num_workers=4)
    val_dataloader = data.DataLoader(val_dataset,
                                    batch_size=batch_size,
                                    shuffle=False,
                                    num_workers=4)
    return train_dataset, val_dataset, train_dataloader, val_dataloader
    



# Dataset for generating the patches given the synthetic cell masks
class SyntheticMaskDataset(data.Dataset):

    def __init__(self, mask_dir, transform=None):
        """Synthetic mask dataset for generating the patches.
        
        Parameters
        ----------
        mask_dir : str
            Path to the folder containing the synthetic masks.
        transform : torchvision.transforms
            Image transformation.
        """
        self.mask_dir = mask_dir
        self.transform = transform
        self.mask_filenames = os.listdir(mask_dir)

    def __len__(self):
        return len(self.mask_filenames)
    
    def __getitem__(self, idx):
        mask_name = self.mask_filenames[idx]
        mask_path = os.path.join(self.mask_dir, mask_name)
        mask = read_image(mask_path)[0, :, :].float()
        if mask.max() > 1:
            mask = mask / 255.0
        return mask.unsqueeze(0)



def prepare_SyntheticMaskDataset(mask_dir,
                                batch_size,
                                transform=None):
        """Prepare the synthetic mask dataset for generating the patches.
        
        Parameters
        ----------
        mask_dir : str
            Path to the folder containing the synthetic masks.
        batch_size : int
            Batch size.
        transform : torchvision.transforms
            Image transformation.
            
        Returns
        -------
        dataset : SyntheticMaskDataset
            Dataset object.
        dataloader : torch.utils.data.DataLoader    
            DataLoader object.
        """

        dataset = SyntheticMaskDataset(mask_dir, transform)
        dataloader = data.DataLoader(dataset, 
                                     batch_size=batch_size, 
                                     shuffle=False, 
                                     num_workers=4)
        return dataset, dataloader

