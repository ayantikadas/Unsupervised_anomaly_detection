import torch
from torch import nn
import sys 
sys.path.insert(0,'/srv/project/APW/Ayantika_codebase/')
from models.encoder_decoder import Conv2DBlock, ConvTranspose2DBlock


class Baur_VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self._conv_layers = self.define_conv_layers()
        self._fully_connected_mu, self._fully_connected_log_var = self.define_fc_layers_enc()
        self._fc_layer = self.define_fc_layer_dec()
        self._convtranspose_layers = self.define_convtranspose_layers()
        
    def _reparameterize(self,mu, log_var):
        """Applying re-parameterization trick."""
        std = torch.exp(0.5 * log_var)  # log_var = log(std^2) = 2*log(std) -> std = exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def define_fc_layers_enc(self):
        fully_connected_mu = nn.Linear(1024, 128)
        fully_connected_log_var = nn.Linear(1024, 128)
        return fully_connected_mu, fully_connected_log_var
    
    def define_fc_layer_dec(self):
        return nn.Linear(128, 1024)
    
    
    def define_conv_layers(self):
        dropout_rate = None
        conv1 = Conv2DBlock(in_channels=1,
                            out_channels=32,
                            conv_further_kwargs={'kernel_size': 5, 'stride': 2, 'padding': 2},
                            dropout_rate=dropout_rate)
        conv2 = Conv2DBlock(in_channels=32,
                            out_channels=64,
                            conv_further_kwargs={'kernel_size': 5, 'stride': 2, 'padding': 2},
                            dropout_rate=dropout_rate)
        conv3 = Conv2DBlock(in_channels=64,
                            out_channels=128,
                            conv_further_kwargs={'kernel_size': 5, 'stride': 2, 'padding': 2},
                            dropout_rate=dropout_rate)
        conv4 = Conv2DBlock(in_channels=128,
                            out_channels=128,
                            conv_further_kwargs={'kernel_size': 5, 'stride': 2, 'padding': 2},
                            dropout_rate=dropout_rate)
        conv5 = Conv2DBlock(in_channels=128,
                            out_channels=16,
                            conv_further_kwargs={'kernel_size': 1, 'stride': 1},
                            dropout_rate=dropout_rate)
        return nn.Sequential(*[conv1, conv2, conv3, conv4, conv5])
    
    def define_convtranspose_layers(self):
        dropout_rate = None
        conv1 = Conv2DBlock(in_channels=16,
                            out_channels=128,
                            conv_further_kwargs={'kernel_size': 1, 'stride': 1},
                            dropout_rate=dropout_rate)
        conv2 = ConvTranspose2DBlock(in_channels=128,
                                     out_channels=128,
                                     conv_further_kwargs={'kernel_size': 5, 'stride': 2,
                                                          'padding': 2, 'output_padding': 1},
                                     dropout_rate=dropout_rate)
        conv3 = ConvTranspose2DBlock(in_channels=128,
                                     out_channels=64,
                                     conv_further_kwargs={'kernel_size': 5, 'stride': 2,
                                                          'padding': 2, 'output_padding': 1},
                                     dropout_rate=dropout_rate)
        conv4 = ConvTranspose2DBlock(in_channels=64,
                                     out_channels=32,
                                     conv_further_kwargs={'kernel_size': 5, 'stride': 2,
                                                          'padding': 2, 'output_padding': 1},
                                     dropout_rate=dropout_rate)
        conv5 = ConvTranspose2DBlock(in_channels=32,
                                     out_channels=32,
                                     conv_further_kwargs={'kernel_size': 5, 'stride': 2,
                                                          'padding': 2, 'output_padding': 1},
                                     dropout_rate=dropout_rate)
        conv6 = Conv2DBlock(in_channels=32,
                            out_channels=1,
                            conv_further_kwargs={'kernel_size': 1, 'stride': 1},
                            normalization_module=None,
                            activation_module=None,
                            dropout_rate=dropout_rate)
        return nn.Sequential(*[conv1, conv2, conv3, conv4, conv5, conv6])


    def forward(self, input_tensor):
        tensor = self._conv_layers(input_tensor)
        tensor = torch.flatten(tensor, start_dim=1)
        mu = self._fully_connected_mu(tensor)
        log_var = self._fully_connected_log_var(tensor)  
        latent_code = self._reparameterize(mu, log_var)
        flat_tensor = self._fc_layer(latent_code)
        reshaped_tensor = flat_tensor.view(-1, 16, 8, 8)
        reconstructed_tensor = self._convtranspose_layers(reshaped_tensor)
        return reconstructed_tensor,latent_code,mu, log_var