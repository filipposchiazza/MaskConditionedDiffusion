import torch
import torch.nn as nn
import os, pickle
import building_modules as bm



class UNet(nn.Module):

    def __init__(self,
                 input_channels,
                 output_channels,
                 base_channels,
                 channel_multiplier,
                 temb_dim,
                 num_resblocks,
                 has_attention,
                 num_heads,
                 dropout_from_resolution,
                 dropout,
                 downsampling_kernel_dim=2,
                 num_classes=1):
        """UNet model for conditioned and uncoditioned Diffusion models
        
        Parameters
        ----------
        input_channels : int
            Number of input channels.
        output_channels : int
            Number of output channels.
        base_channels : int
            Number of base channels for the Convolutional layers and Residual blocks.
        channel_multiplier : list
            List of integers representing the channel multiplier for each downsampling step.
        temb_dim : int
            Dimension of the time embedding.
        num_resblocks : list    
            List of integers representing the number of residual blocks for each downsampling step.
        has_attention : list
            List of booleans representing the presence of the SelfAttention module for each downsampling step.
        num_heads : int
            Number of heads for the SelfAttention module.
        dropout_from_resolution : int
            Resolution from which to apply dropout.
        dropout : float
            Dropout rate.
        downsampling_kernel_dim : int
            Kernel dimension for the first dowsampling convolution.
        num_classes : int
            Number of classes for the label embedding.
        """
        super(UNet, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.base_channels = base_channels
        self.channel_multiplier = channel_multiplier
        self.temb_dim = temb_dim
        self.num_resblocks = num_resblocks
        self.has_attention = has_attention
        self.num_heads = num_heads
        self.dropout_from_resolution = dropout_from_resolution
        self.dropout = dropout
        self.downsampling_kernel_dim = downsampling_kernel_dim
        self.num_classes = num_classes

        # Time embedding
        self.time_emb = bm.TimeEmbedding(dim=temb_dim)
        self.time_mlp = bm.TimeMLP(input_dim=temb_dim, units=temb_dim)

        # Label embedding
        if num_classes > 1:
            self.label_emb = nn.Embedding(self.num_classes, temb_dim)

        # First convolution to reduce the image dimensionality
        self.conv0 = nn.Conv2d(in_channels=input_channels,
                               out_channels=base_channels,
                               kernel_size=downsampling_kernel_dim,
                               stride=downsampling_kernel_dim,
                               padding=0,
                               bias=False)

        # DownBlock
        self.downblock = nn.ModuleList()        
        for i in range(len(self.channel_multiplier)):
            ch = self.base_channels * self.channel_multiplier[i]
            for _ in range(self.num_resblocks[i]):
                self.downblock.append(bm.ResidualBlock(input_channels=ch, 
                                                    t_dim=self.temb_dim, 
                                                    num_channels=ch,
                                                    groups=16))
                if has_attention[i] == True:
                    self.downblock.append(bm.SelfAttention(channels=ch,
                                                        num_heads=self.num_heads,
                                                        dropout=self.dropout))
               
                self.downblock.append(nn.Identity()) # placeholder for the skip
                
            if i != len(self.channel_multiplier) - 1:
                ch_next = self.base_channels * self.channel_multiplier[i+1]
                self.downblock.append(bm.DownSample(in_channels=ch, 
                                                 out_channels=ch_next))
                self.downblock.append(nn.Identity()) # placeholder for the skip


        # MiddleBlock
        ch = self.base_channels * self.channel_multiplier[-1]

        self.middle_res1 = bm.ResidualBlock(input_channels=ch,
                                         t_dim=self.temb_dim,
                                         num_channels=ch,
                                         groups=16)
        
        self.middle_att = bm.SelfAttention(channels=ch,
                                        num_heads=self.num_heads,
                                        dropout=self.dropout)
        
        self.middle_res2 = bm.ResidualBlock(input_channels=ch,
                                         t_dim=self.temb_dim,
                                         num_channels=ch,
                                         groups=16)


        # UpBlock
        self.upblock = nn.ModuleList()
        for i in reversed(range(len(self.channel_multiplier))):
            ch = self.base_channels * self.channel_multiplier[i]
            for _ in range(self.num_resblocks[i] + 1):
                self.upblock.append(nn.Identity()) # placeholder for the concat
                self.upblock.append(bm.ResidualBlock(input_channels=ch*2,
                                                  t_dim=self.temb_dim,
                                                  num_channels=ch))
                if has_attention[i] == True:
                    self.upblock.append(bm.SelfAttention(channels=ch,
                                                      num_heads=self.num_heads,
                                                      dropout=self.dropout))
            if i != 0:
                ch_next = self.base_channels * self.channel_multiplier[i-1]
                self.upblock.append(bm.UpSample(in_channels=ch,
                                            out_channels=ch_next))
        
        # endblock
        self.end_conv = nn.ConvTranspose2d(in_channels=self.base_channels,
                                           out_channels=self.output_channels,
                                           kernel_size=downsampling_kernel_dim,
                                           stride=downsampling_kernel_dim,
                                           padding=0,
                                           bias=False)
        
        # Calculate the number of parameters
        self.num_parameters = self._calculate_num_parameters()


    def forward(self, img, t, mask=None, label=None):
        
        # if a mask is provided, stack it to the input image in the channel dimension
        if mask is not None:
            img = torch.cat((img, mask), dim=1)

        # time embedding
        temb = self.time_emb(t)
        temb = self.time_mlp(temb)

        # label embedding
        if self.num_classes > 1:
            temb_label = self.label_emb(label)
            temb = temb + temb_label

        # First convolution
        x = self.conv0(img)
        skips = [x]
        
        # downblock
        for i in range(len(self.downblock)):
            if type(self.downblock[i]) == nn.Identity:
                skips.append(x)
            elif type(self.downblock[i]) == bm.ResidualBlock:
                x = self.downblock[i]((x, temb))
            else:
                x = self.downblock[i](x)

        # middleblock
        x = self.middle_res1((x, temb))
        x = self.middle_att(x)
        x = self.middle_res2((x, temb))
        
        # upblock
        for i in range(len(self.upblock)):
            
            if type(self.upblock[i]) == nn.Identity:
                x = torch.cat((x, skips.pop()), dim=1)
            elif type(self.upblock[i]) == bm.ResidualBlock:
                x = self.upblock[i]((x, temb))
            else:
                x = self.upblock[i](x)
                
        # endblock
        x = self.end_conv(x)
        
        return x
    


    def _calculate_num_parameters(self):
        """Evaluate the number of trainable and non-trainable model parameters

        Returns
        -------
        p_train : int
            Number of trainable parameters.
        p_non_train : int
            Number of non-trainable parameters.
        """
        p_total = 0
        p_train = 0
        for p in self.parameters():
            p_total += p.numel()
            if p.requires_grad:
                p_train += p.numel()       
        p_non_train = p_total - p_train
        return p_train, p_non_train



    def save_model(self, save_folder):
        """Save the parameters and the model state_dict
        
        Parameters:
        ----------
        save_folder: str
            Folder to save the model
        """
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        
        param_file = os.path.join(save_folder, 'UnetParameters.pkl')
        parameters = [self.input_channels,
                      self.output_channels,
                      self.base_channels,
                      self.channel_multiplier,
                      self.temb_dim,
                      self.num_resblocks,
                      self.has_attention,
                      self.num_heads,
                      self.dropout_from_resolution,
                      self.dropout,
                      self.downsampling_kernel_dim]

        with open(param_file, 'wb') as f:
            pickle.dump(parameters, f)
    
        model_file = os.path.join(save_folder, 'UnetModel.pt')
        torch.save(self.state_dict(), model_file)


    
    @staticmethod
    def save_history(history, save_folder):
        """Save the training history
        
        Parameters
        ----------
        history : dict
            Training history.
        save_folder : str
            Path to the folder where to save the training and validation history.
            
        Returns
        -------
        None."""
        filename = os.path.join(save_folder, 'diffusion_history.pkl')
        with open(filename, 'wb') as f:
            pickle.dump(history, f)
    


    @classmethod
    def load_model(cls, save_folder, swa_version=False):
        """Load the parameters and the model state_dict
        
        Parameters:
        ----------
        save_folder: str
            Folder to load the model from
        swa_version: bool
            If True, load the SWA model

        Returns
        -------
        model : UNet
            UNet model with the loaded parameters and state_dict
        """
        param_file = os.path.join(save_folder, 'UnetParameters.pkl') 
        with open(param_file, 'rb') as f:
            parameters = pickle.load(f)
        
        model = cls(*parameters)

        if swa_version == True:
            model_file = os.path.join(save_folder, 'SWAModel.pt')
        else:
            model_file = os.path.join(save_folder, 'UnetModel.pt')
            
        model.load_state_dict(torch.load(model_file, map_location='cuda:0'))
    
        return model
    


    @staticmethod
    def load_history(save_folder):
        """Load the training history
            
        Parameters
        ----------
        save_folder : str
            Path to the folder where the training history is saved.
        
        Returns
        -------
        history : dict
            Training and validation history.
        """
        history_file = os.path.join(save_folder, 'diffusion_history.pkl')
        with open(history_file, 'rb') as f:
            history = pickle.load(f)
        return history


