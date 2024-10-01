import torch
import matplotlib.pyplot as plt
from dataset import prepare_ImageDataset
import config
from unet import UNet
from gaussian_diffusion_utils import GaussianDiffusion
import pickle
import os
import random
from tqdm import tqdm



# Load image dataset (only the validation masks will be used for generation)
train_dataset, val_dataset, train_dataloader, val_dataloader = prepare_ImageDataset(img_dir=config.IMG_DIR,
                                                                                    batch_size=config.GEN_BATCH_SIZE,
                                                                                    validation_split=config.VALIDATION_SPLIT,
                                                                                    transform=config.TRANSFORM,
                                                                                    seed=123,
                                                                                    fraction=config.FRACTION,
                                                                                    normalization_mode=config.NORMALIZATION_MODE)


# Load model
model = UNet.load_model(config.MODEL_DIR, swa_version=True).to(config.DEVICE)

# Load Gaussian diffusion utility
gdf_util = GaussianDiffusion(schedule='cosine_shifted',
                             timesteps=config.TIMESTEPS,
                             beta_start=config.BETA_START,
                             beta_end=config.BETA_END,
                             clip_min=config.CLIP_MIN,
                             clip_max=config.CLIP_MAX,
                             img_size=config.IMG_DIM)


# Generate images from real masks
save_counter = 1
for i, (imgs, masks) in enumerate(tqdm(val_dataloader)):
    if i == config.NUM_ITERATION:
        break
    masks = masks.to(config.DEVICE)
    gen_imgs = gdf_util.generate_sample_from_masks(model, masks)

    for j in range(config.GEN_BATCH_SIZE):
        img = gen_imgs[j].detach().cpu().numpy().transpose(1, 2, 0)
        plt.imsave(os.path.join(config.SAVE_SYN_FOLDER, f'gen_img_{save_counter}.png'), img)
        save_counter += 1



