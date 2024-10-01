import torch
import torch.optim as optim
import config
from dataset import prepare_ImageDataset
from custom_lr_scheduler import custom_lr_schedule_wrapper
import unet as unet
from gaussian_diffusion_utils import GaussianDiffusion
from diffusion_trainer import DiffusionTrainer
import os


# Load image dataset
train_dataset, val_dataset, train_dataloader, val_dataloader = prepare_ImageDataset(img_dir=config.IMG_DIR,
                                                                                    batch_size=config.BATCH_SIZE,
                                                                                    validation_split=config.VALIDATION_SPLIT,
                                                                                    transform=config.TRANSFORM,
                                                                                    seed=123,
                                                                                    fraction=config.FRACTION,
                                                                                    normalization_mode=config.NORMALIZATION_MODE)
# Create Unet and Gdf_util
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

# Create SWA model
swa_model = optim.swa_utils.AveragedModel(model)

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

# Train
history = trainer.train(train_dataloader=train_dataloader,
                        num_epochs=config.NUM_EPOCHS,
                        save_folder=config.SAVE_FOLDER,
                        val_dataloader=val_dataloader,
                        save_checkpoints=True,
                        swa_update_epochs=config.SWA_UPDATE_EPOCHS,
                        grad_clip=config.GRAD_CLIP)


# Save model, Gdf_util, history and swa_model
model.save_model(config.SAVE_FOLDER)
gdf_util.save(config.SAVE_FOLDER)
model.save_history(history, config.SAVE_FOLDER)

swa_model_file = os.path.join(config.SAVE_FOLDER, 'SWAModel.pt')
torch.save(swa_model.module.state_dict(), swa_model_file)

