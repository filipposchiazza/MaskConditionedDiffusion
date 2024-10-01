import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import os


class DiffusionTrainer():

    def __init__(self, 
                 model,   
                 gdf_util,
                 optimizer,
                 lr_scheduler=None,
                 device='cpu',
                 verbose=True,
                 swa_model=None):

        """Diffusion trainer class.

        Parameters
        ----------
        model : nn.Module
            Diffusion model to be trained.
        gdf_util : GDFUtil
            Gaussian diffusion utility object.
        optimizer : torch.optim.Optimizer
            Optimizer object.
        lr_scheduler : torch.optim.lr_scheduler._LRScheduler, optional
            Learning rate scheduler object, by default None.
        device : str, optional
            Device to use for training, by default 'cpu'.
        verbose : bool, optional
            Verbosity flag, by default True.
        swa_model : nn.Module, optional
            Stochastic weight averaging model, by default None.
        """
        
        self.model = model.to(device)
        self.gdf_util = gdf_util
        self.timesteps = gdf_util.timesteps
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.verbose = verbose
        self.swa_model = swa_model



    def train(self,
              train_dataloader,
              num_epochs,
              save_folder,
              val_dataloader=None,
              save_checkpoints=True,
              swa_update_epochs=[],
              grad_clip=None,
              grad_accum_steps=None):
        
        """Train the diffusion model.

        Parameters
        ----------
        train_dataloader : torch.utils.data.DataLoader
            Training dataloader.
        num_epochs : int
            Number of epochs.
        save_folder : str
            Folder where to save the model checkpoints.
        val_dataloader : torch.utils.data.DataLoader, optional
            Validation dataloader, by default None.
        save_checkpoints : bool, optional
            Flag to save model checkpoints, by default True.
        swa_update_epochs : list, optional
            List of epochs where to update the SWA model, by default [].
        grad_clip : float, optional
            Gradient clipping value, by default None.
        grad_accum_steps : int, optional
            Number of gradient accumulation steps, by default None.

        Returns
        -------
        history : dict
            Training history.
        """
        # Initialize history
        history = {'loss_train': [],
                   'loss_val': []}
        
        # Create checkpoint folder
        os.makedirs(os.path.join(save_folder, 'checkpoints'), exist_ok=True)

        # Take some masks from the validation set to generate images
        mask_sample = next(iter(val_dataloader))[1][:10]
        
        for epoch in range(num_epochs):
            
            # Training mode
            self.model.train()

            # Train one epoch
            train_loss = self._train_one_epoch(train_dataloader=train_dataloader,
                                               epoch=epoch,
                                               grad_clip=grad_clip,
                                               grad_accum_steps=grad_accum_steps)

            # Update history
            history['loss_train'].append(train_loss)

            # Update SWA model (if a SWA model is provided)
            if self.swa_model is not None and epoch in swa_update_epochs:
                self.swa_model.update_parameters(self.model)
                print("SWA model updated, when LR is: ", self.lr_scheduler.get_last_lr()[0])

            # Update learning rate (if a learning rate scheduler is provided)
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            if val_dataloader is not None:
                # Validation mode
                self.model.eval()

                # Validate one epoch
                val_loss = self._validate(val_dataloader=val_dataloader)

                # Update history
                history['loss_val'].append(val_loss)

            
            if epoch != 0 and epoch % 5 == 0:
                
                # Save model checkpoint
                if save_checkpoints == True:
                    os.makedirs(os.path.join(save_folder, 'checkpoints', f'epoch_{epoch}'), exist_ok=True)
                    self.model.save_model(os.path.join(save_folder, 'checkpoints', f'epoch_{epoch}'))

                # Generate some images to visualize the training progress 
                if self.verbose == True:
                    os.makedirs(os.path.join(save_folder, 'checkpoints', f'epoch_{epoch}'), exist_ok=True)
                    gen_imgs = self.gdf_util.generate_sample_from_masks(model=self.model, masks=mask_sample)
                    filepath = os.path.join(save_folder, 'checkpoints', f'epoch_{epoch}', f'gen_imgs_epoch_{epoch}.pt')
                    torch.save(gen_imgs, filepath)
        
        return history



    def _train_one_epoch(self,
                         train_dataloader,
                         epoch,
                         grad_clip,
                         grad_accum_steps):
        """Train one epoch.

        Parameters
        ----------
        train_dataloader : torch.utils.data.DataLoader
            Training dataloader.
        epoch : int
            Current epoch.
        grad_clip : float
            Gradient clipping value.
        grad_accum_steps : int
            Number of gradient accumulation steps.
        
        Returns
        -------
        mean_loss : float
            Mean training loss of the epoch.
        """
        
        
        running_loss = 0.
        mean_loss = 0.

        self.optimizer.zero_grad()

        with tqdm(train_dataloader, unit="batches") as tepoch:

            for batch_idx, data in enumerate(tepoch):

                # Update the progress bar description
                tepoch.set_description(f'Epoch {epoch+1}')

                # Load images and masks to device
                imgs, masks = data
                imgs = imgs.to(device=self.device, dtype=torch.float)
                masks = masks.to(device=self.device, dtype=torch.float)

                # 1. Get the batch size
                batch_size = imgs.shape[0]
        
                # 2. Sample timesteps uniformely
                t = torch.randint(low=0, high=self.timesteps, size=(batch_size, )).to(self.device)
        
                # 3. Sample random noise to be added to the images in the batch
                noise = torch.randn(size=imgs.shape, dtype=imgs.dtype).to(self.device)

                # 4. Diffuse the images with noise
                imgs_t = self.gdf_util.q_sample(imgs, t, noise).to(self.device, dtype=torch.float)
        
                # 5. Pass the diffused images and time steps to the unet
                pred_noise = self.model(imgs_t, t, masks)
                
                # 6. Calculate the loss 
                loss = F.mse_loss(noise, pred_noise)
                loss.backward()

                # 7. Gradient clipping
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)

                # 8. Update the weights and zero the gradients, eventually with gradient accumulation
                if grad_accum_steps is None or (batch_idx + 1) % grad_accum_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
                # 9. Update running losses and mean losses
                running_loss += loss.item()
                mean_loss = running_loss / (batch_idx + 1)
            
                tepoch.set_postfix(loss="{:.6f}".format(mean_loss))

            # Handle remaining gradients if the number of batches is not divisible by accumulation_steps
            if grad_accum_steps is not None and (batch_idx + 1) % grad_accum_steps != 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
        
        return mean_loss
    


    def _validate(self, val_dataloader):
        """Validate one epoch.

        Parameters
        ----------
        val_dataloader : torch.utils.data.DataLoader
            Validation dataloader.

        Returns
        -------
        mean_val_loss : float
            Mean validation loss of the epoch.
        """

        running_val_loss = 0.

        with torch.no_grad():
            for batch_idx, data in enumerate(val_dataloader):

                # Load images and masks to device
                imgs, masks = data
                imgs = imgs.to(device=self.device, dtype=torch.float)
                masks = masks.to(device=self.device, dtype=torch.float)

                # 1. Get the batch size
                batch_size = imgs.shape[0]
        
                # 2. Sample timesteps uniformely
                t = torch.randint(low=0, high=self.timesteps, size=(batch_size, )).to(self.device)
        
                # 3. Sample random noise to be added to the images in the batch
                noise = torch.randn(size=imgs.shape, dtype=imgs.dtype).to(self.device)

                # 4. Diffuse the images with noise
                imgs_t = self.gdf_util.q_sample(imgs, t, noise).to(self.device, dtype=torch.float)
        
                # 5. Pass the diffused images and time steps to the unet
                pred_noise = self.model(imgs_t, t, masks)
                
                # 6. Calculate the loss 
                loss = F.mse_loss(noise, pred_noise)
            
                # 7. Update running losses and mean losses
                running_val_loss += loss.item()

        mean_val_loss = running_val_loss / len(val_dataloader)

        if self.verbose == True:
            print(f"Validation loss: {mean_val_loss:.6f}")

        return mean_val_loss
    
