import matplotlib.pyplot as plt
from dataset import prepare_SyntheticMaskDataset
import config
from unet import UNet
from gaussian_diffusion_utils import GaussianDiffusion
import os
from tqdm import tqdm


# Load synthetic mask dataset
dataset, dataloader = prepare_SyntheticMaskDataset(mask_dir=config.SYNTHETIC_MASK_DIR,
                                                   batch_size=config.GEN_BATCH_SIZE)

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
for i, mask in enumerate(tqdm(dataloader)):
    if i == config.NUM_ITERATION:
        break
    mask = mask.to(config.DEVICE)
    gen_imgs = gdf_util.generate_sample_from_masks(model, mask)

    for j in range(config.GEN_BATCH_SIZE):
        img = gen_imgs[j].detach().cpu().numpy().transpose(1, 2, 0)
        plt.imsave(os.path.join(config.SAVE_SYN_FOLDER, f'gen_img_{save_counter}.png'), img)
        save_counter += 1