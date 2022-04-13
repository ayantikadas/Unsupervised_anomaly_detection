import os
import copy
import glob
import torch
import numpy as np
import time

import torch.nn as nn
from tqdm.notebook import tqdm
import torch.nn.functional as F
from torchsummary import summary
from monai.data import CacheDataset
from torch.utils.data import Dataset
from scipy.ndimage import gaussian_filter

class custom_VAE(nn.Module):
    def conv2d_block(self, in_channels, out_channels, kernel_size = 3,stride = 2,padding = 2,norm_ = True, activation_ = True):
        
        
        if norm_ and activation_:        
            block = torch.nn.Sequential(
                        torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels,stride = stride,padding = padding),
                        torch.nn.InstanceNorm2d(out_channels),                        
                        torch.nn.PReLU(),
                
                    )
        elif norm_== True and activation_ == False :
            block = torch.nn.Sequential(
                        torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels,stride = stride,padding = padding),
                        torch.nn.InstanceNorm2d(out_channels),
                    )
        elif norm_== False and activation_== True:
            block = torch.nn.Sequential(
                        torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels,stride = stride,padding = padding),
                        torch.nn.PReLU(),
                    )
        else:
            block = torch.nn.Sequential(
                        torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels,stride = stride,padding = padding),

                    )
        return block
    
    def ff_block(self,in_channels, intr_channel,z_dim):
        
        block = torch.nn.Sequential(
                    torch.nn.Linear(in_channels, intr_channel),                 
                    torch.nn.PReLU(),
                    torch.nn.Linear( intr_channel, z_dim*2),
                )
        
        return block
    
    def convTr2d_block(self, in_channels, out_channels, kernel_size = 3,stride = 2,padding = 1,output_padding= 1, norm_ = True, activation_ = True):
        if norm_ and activation_:        
            block = torch.nn.Sequential(
                torch.nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding,output_padding = output_padding),
                torch.nn.InstanceNorm2d(out_channels), ##, affine=True
                torch.nn.PReLU(),
                )
        
        elif norm_== True and activation_ == False :
            block = torch.nn.Sequential(
                torch.nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding,output_padding = output_padding),
                torch.nn.InstanceNorm2d(out_channels), ##, affine=True
                )
        elif norm_== False and activation_== True:

            block = torch.nn.Sequential(
                torch.nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding,output_padding = output_padding),
                torch.nn.PReLU(),
                )
        else:
            block = torch.nn.Sequential(
                torch.nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding,output_padding = output_padding),
                )
        return block
    
    def reparametrize(self,mu, logvar):
        std = logvar.div(2).exp()
        #Variable
        eps = std.data.new(std.size()).normal_()
        return mu + std*eps

        
    
    def __init__(self, encode_layers_info,decode_layers_info,\
                 input_height,input_width,\
                 z_dim,intr_channel):
        super(custom_VAE, self).__init__()
        self.input_height = input_height
        self.input_width = input_width
        self.intr_channel = intr_channel
        self.z_dim = z_dim
        self.encode_layers_info = encode_layers_info
        self.decode_layers_info = decode_layers_info
#         print(self.input_height,self.input_width)
        '''
        Description of arguments
        encode_layers_info = {'layers_desc':'5 encoding layers','layers':5,\
                              'input_channels':[1,32,64,64,128],\
                              'output_channels':[32,64,64,128,128],\
                              'kernel_size':[3,3,3,3,3],\
                              'stride': [2]*5,'padding': [1]*5,\
                              'norm_':[True]*5,'activation_':[True]*5
                             }

        decode_layers_info = {'layers_desc':'5 decoding layers','layers':5,\
                              'input_channels':[128,128,64,64,32],\
                              'output_channels':[128,64,64,32,1],\
                              'kernel_size':[3,3,3,3],\
                              'stride': [2]*5,'padding': [1]*5,\
                              'output_padding': [1]*5,\
                              'norm_':[True]*5,'activation_':[True]*5
                             }
        '''
        
        
        #Encode
        encode_bt_layer_list = []
        for layer_ in range(self.encode_layers_info['layers']):
            encode_bt_layer_list.append(self.conv2d_block(self.encode_layers_info['input_channels'][layer_],\
                                              self.encode_layers_info['output_channels'][layer_], \
                                              self.encode_layers_info['kernel_size'][layer_],\
                                              self.encode_layers_info['stride'][layer_],\
                                              self.encode_layers_info['padding'][layer_],\
                                              self.encode_layers_info['norm_'][layer_], \
                                              self.encode_layers_info['activation_'][layer_]))
            
            
        decode_bt_layer_list = []
        for layer_ in range(self.decode_layers_info['layers']):
            decode_bt_layer_list.append(self.convTr2d_block(self.decode_layers_info['input_channels'][layer_], \
                                                          self.decode_layers_info['output_channels'][layer_], \
                                                          self.decode_layers_info['kernel_size'][layer_],\
                                                          self.decode_layers_info['stride'][layer_],\
                                                          self.decode_layers_info['padding'][layer_],\
                                                          self.decode_layers_info['output_padding'][layer_],\
                                                          self.decode_layers_info['norm_']\
                                                          [layer_], self.decode_layers_info['activation_'][layer_]))

        
        # Encode
        self.conv_encode1 = encode_bt_layer_list[0]
        self.conv_encode2 = encode_bt_layer_list[1]
        self.conv_encode3 = encode_bt_layer_list[2]
        self.conv_encode4 = encode_bt_layer_list[3]
        self.conv_encode5 = encode_bt_layer_list[4]
        self.no_linear_nodes = self.encode_layers_info['output_channels'][-1]*\
                                (round(self.input_height/(2**self.encode_layers_info['layers'])))*\
                                (round(self.input_width/(2**self.encode_layers_info['layers'])))
        
        # Bottleneck
        self.bottleneck_layer= self.ff_block(
                            in_channels = self.no_linear_nodes,\
                            intr_channel= self.intr_channel,z_dim = self.z_dim)        
        
        # Decode
        self.post_bottleneck_layer = self.ff_block(in_channels = self.z_dim, \
                                              intr_channel= self.intr_channel,\
                                              z_dim = (self.no_linear_nodes)//2
                                             )
        self.conv_decode5 = decode_bt_layer_list[0]
        self.conv_decode4 = decode_bt_layer_list[1]
        self.conv_decode3 = decode_bt_layer_list[2]
        self.conv_decode2 = decode_bt_layer_list[3]
        self.conv_decode1  = decode_bt_layer_list[4]
    
    def forward(self,x):
        # Encode
        encode_block1 = self.conv_encode1(x)
        encode_block2 = self.conv_encode2(encode_block1)
        encode_block3 = self.conv_encode3(encode_block2)
        encode_block4 = self.conv_encode4(encode_block3)
        encode_block5 = self.conv_encode5(encode_block4)
        # Bottleneck
        bt = self.bottleneck_layer(encode_block5.view(-1,self.no_linear_nodes))
        # reparameterization
        reparametrize_layer = self.reparametrize(bt[:,:self.z_dim], bt[:,self.z_dim:])
        
        # Decode
        cat_layer5 = self.post_bottleneck_layer(reparametrize_layer)   
        cat_layer4 = self.conv_decode5(cat_layer5.view(-1,self.encode_layers_info['output_channels'][-1],
                                (round(self.input_height/(2**self.encode_layers_info['layers']))),
                                (round(self.input_width/(2**self.encode_layers_info['layers'])))))
        cat_layer3 = self.conv_decode4(cat_layer4)
        cat_layer2 = self.conv_decode3(cat_layer3)
        cat_layer1 = self.conv_decode2(cat_layer2)
        x_recon = self.conv_decode1(cat_layer1)
        return x_recon,bt[:,:self.z_dim],bt[:,self.z_dim:]
        
