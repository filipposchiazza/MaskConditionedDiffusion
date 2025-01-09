import matplotlib.pyplot as plt
import config
from unet import UNet
from gaussian_diffusion_utils import GaussianDiffusion
import os
from tqdm import tqdm
import torch




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
save_counter = config.IDX_START
gen_4_channels = True

for i in tqdm(range(config.NUM_ITERATION)):
    gen_imgs = gdf_util.generate_sample_uncoditionally(model=model, num_samples=config.GEN_BATCH_SIZE, sample_channels=4)
    for j in range(config.GEN_BATCH_SIZE):

        if gen_4_channels == True:
            sample = gen_imgs[j].detach().cpu()
            img = sample[:3].permute(1, 2, 0)
            mask = sample[3].unsqueeze(0).repeat(3, 1, 1).permute(1, 2, 0)
            full_img = torch.cat((img, mask), dim=1).numpy()
            plt.imsave(os.path.join(config.SAVE_FOUR_CHANNELS, f'gen_img_{save_counter}.png'), full_img)
        
        else:
            img = gen_imgs[j].detach().cpu().numpy().transpose(1, 2, 0)
            plt.imsave(os.path.join(config.SAVE_FOUR_CHANNELS, f'gen_img_{save_counter}.png'), img)
        
        save_counter += 1



