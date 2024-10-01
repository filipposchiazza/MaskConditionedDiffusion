# Diffusion for High Resolution Images with Segmentation-based conditioning
This repository is based on the paper "Simple diffusion: End-to-end diffusion for high resolution images" (https://arxiv.org/abs/2301.11093) and it is applied to the field of medical imaging generation. This project add the possibility to condition the diffusion process with a segmentation mask. 

## Repository Structure
The files are organized as follows:
- `train.py`: Contains an example script to train the Diffusion model.
- `diffusion_trainer.py`: Contains the class definition for the Diffusion Trainer, that is the what will handle the training process.
- `unet.py`: Contains the implementation of the Unet model.
- `dataset.py`: Contains the implementation of the dataset classes.
- `building_modules.py`: Contains the implementation of the building blocks for the Diffusion model.
- `gaussian_diffusion_utils.py`: Contains the implementation of the Gaussian Diffusion process.
- `config.py`: Contains the configuration parameters for the Diffusion model and the training process.
- `gen_from_real_masks.py` and `gen_from_synthetic_masks.py`: Contains the scripts to generate the images from the segmentation masks.
- `custom_lr_scheduler.py`: Contains the implementation of the custom learning rate scheduler.

## How to use for training
Import the necessary dependencies:
```python
import torch
import torch.optim as optim
import config
from dataset import prepare_ImageDataset
from custom_lr_scheduler import custom_lr_schedule_wrapper
import unet as unet
from gaussian_diffusion_utils import GaussianDiffusion
from diffusion_trainer import DiffusionTrainer
import os
```

Load the dataset:
```python
train_dataset, val_dataset, train_dataloader, val_dataloader = prepare_ImageDataset(img_dir=config.IMG_DIR,
                                                                                    batch_size=config.BATCH_SIZE,
                                                                                    validation_split=config.VALIDATION_SPLIT,
                                                                                    transform=config.TRANSFORM,
                                                                                    seed=123,
                                                                                    fraction=config.FRACTION,
                                                                                    normalization_mode=config.NORMALIZATION_MODE)
```

Create the Unet model and the Gaussian Diffusion process:
```python
gdf_util = GaussianDiffusion(schedule=config.SCHEDULE,
                             timesteps=config.TIMESTEPS,
                             beta_start=config.BETA_START,
                             beta_end=config.BETA_END,
                             clip_min=config.CLIP_MIN,
                             clip_max=config.CLIP_MAX,
                             img_size=config.IMG_DIM)


model = unet.UNet(input_channels=config.INPUT_CHANNELS,
                  output_channels=config.OUTPUT_CHANNELS,
                  base_channels=config.BASE_CHANNELS,
                  channel_multiplier=config.CHANNEL_MULTIPLIER,
                  temb_dim=config.TEMB_DIM,
                  num_resblocks=config.NUM_RES_BLOCKS,
                  has_attention=config.HAS_ATTENTION,
                  num_heads=config.NUM_HEADS,
                  dropout_from_resolution=config.DROPOUT_FROM_RESOLUTION,
                  dropout=config.DROPOUT,
                  downsampling_kernel_dim=config.DOWNSAMPLING_KERNEL_DIM)
```

Eventually, create also the Sthocastic Weight Averaging (SWA) model:
```python
swa_model = optim.swa_utils.AveragedModel(model)
```

Create the optimizer, the learning rate scheduler and the trainer:
```python
# Create optimizer
optimizer = optim.Adam(model.parameters(), lr=1.0)  # If you want to use a custom learning rate scheduler, set lr=1.0

# Create custom learning rate scheduler
lambda_fn = custom_lr_schedule_wrapper(num_epochs=config.NUM_EPOCHS,
                                       lr_start=config.LR_START,
                                       warmup_step=config.WARMUP_STEP,
                                       warmup_lr_end=config.WARMUP_LR_END,
                                       annealing_step=config.ANNEALING_STEP,
                                       annealing_lr_end=config.ANNEALING_LR_END,
                                       num_cicles=config.NUM_CYCLES,
                                       max_cicles_lr=config.MAX_CYCLES_LR)

scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_fn)

# Create trainer
trainer = DiffusionTrainer(model=model,
                           gdf_util=gdf_util,
                           optimizer=optimizer,
                           lr_scheduler=scheduler,
                           device=config.DEVICE,
                           verbose=True,
                           swa_model=swa_model)
```

Train the model:
```python
history = trainer.train(train_dataloader=train_dataloader,
                        num_epochs=config.NUM_EPOCHS,
                        save_folder=config.SAVE_FOLDER,
                        val_dataloader=val_dataloader,
                        save_checkpoints=True,
                        swa_update_epochs=config.SWA_UPDATE_EPOCHS,
                        grad_clip=config.GRAD_CLIP)
```

Save the model, the swa_model and the history:
```python
model.save_model(config.SAVE_FOLDER)
gdf_util.save(config.SAVE_FOLDER)
model.save_history(history, config.SAVE_FOLDER)

swa_model_file = os.path.join(config.SAVE_FOLDER, 'SWAModel.pt')
torch.save(swa_model.module.state_dict(), swa_model_file)
```


## How to use for generation
Import the necessary dependencies:
```python
import matplotlib.pyplot as plt
from dataset import prepare_SyntheticMaskDataset
import config
from unet import UNet
from gaussian_diffusion_utils import GaussianDiffusion
import os
from tqdm import tqdm
```

Load the synthetic mask dataset:
```python
dataset, dataloader = prepare_SyntheticMaskDataset(mask_dir=config.SYNTHETIC_MASK_DIR,
                                                   batch_size=config.GEN_BATCH_SIZE)
```

Load the model and the Gaussian Diffusion utility:
```python
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
```

Generate the images from the synthetic masks and save them:
```python
# Generate images from real masks
save_counter = 1
for i, mask in enumerate(tqdm(dataloader)):
    if i == config.NUM_ITERATION:
        break
    mask = mask.to(config.DEVICE)
    gen_imgs = gdf_util.generate_sample_from_masks(model, mask)

    for j in range(config.GEN_BATCH_SIZE):
        img = gen_imgs[j].detach().cpu().numpy().transpose(1, 2, 0)
        plt.imsave(os.path.join(config.SAVE_SEMI_SYN_FOLDER, f'gen_img_{save_counter}.png'), img)
        save_counter += 1
```

## Dependencies


