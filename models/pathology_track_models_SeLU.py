import os
import copy
import glob
import torch
import numpy as np
import time

import torch.nn as nn
import torchio as tio
from torchviz import make_dot
from tqdm.notebook import tqdm
import torch.nn.functional as F
from torchsummary import summary
from monai.data import CacheDataset
from torch.utils.data import Dataset
from scipy.ndimage import gaussian_filter

# Abridged version of https://towardsdatascience.com/u-net-b229b32b4a71

# Abridged version of https://towardsdatascience.com/u-net-b229b32b4a71

class custom_AE0(nn.Module):
    def contracting_block(self, in_channels, out_channels, kernel_size):
        block = torch.nn.Sequential(
                    torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels,stride = 2,padding = 2),
                    torch.nn.BatchNorm2d(out_channels),
                    torch.nn.SELU(),
#                     torch.nn.AlphaDropout(p=0.2),
                )
        return block
    
    def expansive_block(self, in_channels, out_channels, kernel_size):
        block = torch.nn.Sequential(
                torch.nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=2, padding=2,output_padding = 1),
                torch.nn.BatchNorm2d(out_channels),
                torch.nn.SELU(),
                )
        return block
    
    def bottleneck_block(self,in_channels, out_channels):
        block = torch.nn.Sequential(
                    torch.nn.Conv2d(kernel_size=1, in_channels=in_channels, out_channels=out_channels,stride = 1,padding = 0),
                    # torch.nn.LeakyReLU(),
                )
        return block
    
    def post_bottleneck_block(self,in_channels, out_channels):
        block = torch.nn.Sequential(
                    torch.nn.Conv2d(kernel_size=1, in_channels=in_channels, out_channels=out_channels,stride = 1,padding = 0),
                    torch.nn.SELU(),
                )
        return block
    
    def final_block(self, in_channels, out_channels, kernel_size):
        block = torch.nn.Sequential(
                torch.nn.ConvTranspose2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, stride=2, padding=2,output_padding = 1),
                torch.nn.SELU(),
                torch.nn.Conv2d(kernel_size=1, in_channels=in_channels, out_channels=out_channels),
#                 torch.nn.Tanh()
                )
        return block
    
    def __init__(self, in_channel, out_channel):
        super(custom_AE0, self).__init__()
        #Encode
        self.conv_encode1 = self.contracting_block(in_channels=in_channel, out_channels=32,kernel_size = 5)
        self.conv_encode2 = self.contracting_block(32,64,5)
        self.conv_encode3 = self.contracting_block(64,128,5)
        self.conv_encode4 = self.contracting_block(128,128,5)
        # Bottleneck
        self.bottleneck_layer = self.bottleneck_block(128,16)
        # Decode
        self.post_bottleneck_layer = self.post_bottleneck_block(16,128)
        self.conv_decode3 = self.expansive_block(128,128,5)
        self.conv_decode2 = self.expansive_block(128,64,5)
        self.conv_decode1 = self.expansive_block(64,32,5)
        self.final_layer = self.final_block(32, out_channel,5)
    
    def forward(self, x):
        # Encode
        encode_block1 = self.conv_encode1(x)
        encode_block2 = self.conv_encode2(encode_block1)
        encode_block3 = self.conv_encode3(encode_block2)
        encode_block4 = self.conv_encode4(encode_block3)
        # Bottleneck
        bt = self.bottleneck_layer(encode_block4)
        # Decode
        cat_layer4 = self.post_bottleneck_layer(bt)
        cat_layer3 = self.conv_decode3(cat_layer4)
        cat_layer2 = self.conv_decode2(cat_layer3)
        cat_layer1 = self.conv_decode1(cat_layer2)
        final_layer = self.final_layer(cat_layer1)
        return final_layer,bt
    
# Abridged version of https://towardsdatascience.com/u-net-b229b32b4a71

class custom_AE1(nn.Module):
    def contracting_block(self, in_channels, out_channels, kernel_size):
        block = torch.nn.Sequential(
                    torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels,stride = 2,padding = 2),
                    torch.nn.BatchNorm2d(out_channels),
                    torch.nn.LeakyReLU(),
                )
        return block
    
    def expansive_block(self, in_channels, out_channels, kernel_size):
        block = torch.nn.Sequential(
                torch.nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=2, padding=2,output_padding = 1),
                torch.nn.BatchNorm2d(out_channels),
                torch.nn.LeakyReLU(),
                )
        return block
    
    def bottleneck_block(self,in_channels, out_channels):
        block = torch.nn.Sequential(
                    torch.nn.Conv2d(kernel_size=1, in_channels=in_channels, out_channels=out_channels,stride = 1,padding = 0),
                    # torch.nn.LeakyReLU(),
                )
        return block
    
    def post_bottleneck_block(self,in_channels, out_channels):
        block = torch.nn.Sequential(
                    torch.nn.Conv2d(kernel_size=1, in_channels=in_channels, out_channels=out_channels,stride = 1,padding = 0),
                    torch.nn.LeakyReLU(),
                )
        return block
    
    def final_block(self, in_channels, out_channels, kernel_size):
        block = torch.nn.Sequential(
                torch.nn.ConvTranspose2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, stride=2, padding=2,output_padding = 1),
                torch.nn.LeakyReLU(),
                torch.nn.Conv2d(kernel_size=1, in_channels=in_channels, out_channels=out_channels),
#                 torch.nn.Tanh()
                )
        return block
    
    def __init__(self, in_channel, out_channel):
        super(custom_AE1, self).__init__()
        #Encode
        self.conv_encode1 = self.contracting_block(in_channels=in_channel, out_channels=32,kernel_size = 5)
        self.conv_encode2 = self.contracting_block(32,64,5)
        self.conv_encode3 = self.contracting_block(64,128,5)
        self.conv_encode4 = self.contracting_block(128,128,5)
        # Bottleneck
        self.bottleneck_layer = self.bottleneck_block(128,16)
        # Decode
        self.post_bottleneck_layer = self.post_bottleneck_block(16,128)
        self.conv_decode3 = self.expansive_block(128,128,5)
        self.conv_decode2 = self.expansive_block(128,64,5)
        self.conv_decode1 = self.expansive_block(64,32,5)
        self.final_layer = self.final_block(32, out_channel,5)
    
    def forward(self, x):
        # Encode
        encode_block1 = self.conv_encode1(x)
        encode_block2 = self.conv_encode2(encode_block1)
        encode_block3 = self.conv_encode3(encode_block2)
        encode_block4 = self.conv_encode4(encode_block3)
        # Bottleneck
        bt = self.bottleneck_layer(encode_block4)
        # Decode
        cat_layer4 = self.post_bottleneck_layer(bt)
        cat_layer3 = self.conv_decode3(cat_layer4)
        cat_layer2 = self.conv_decode2(cat_layer3)
        cat_layer1 = self.conv_decode1(cat_layer2)
        final_layer = self.final_layer(cat_layer1)
        return final_layer
# Abridged version of https://towardsdatascience.com/u-net-b229b32b4a71

class custom_AE2(nn.Module):
    def contracting_block(self, in_channels, out_channels, kernel_size):
        block = torch.nn.Sequential(
                    torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels,stride = 2,padding = 2),
                    torch.nn.BatchNorm2d(out_channels),
                    torch.nn.LeakyReLU(),
                )
        return block
    
    def expansive_block(self, in_channels, out_channels, kernel_size):
        block = torch.nn.Sequential(
                torch.nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=2, padding=2,output_padding = 1),
                torch.nn.BatchNorm2d(out_channels),
                torch.nn.LeakyReLU(),
                )
        return block
    
    def bottleneck_block(self,in_channels, out_channels):
        block = torch.nn.Sequential(
                    torch.nn.Conv2d(kernel_size=1, in_channels=in_channels, out_channels=out_channels,stride = 1,padding = 0),
                    # torch.nn.LeakyReLU(),
                )
        return block
    
    def post_bottleneck_block(self,in_channels, out_channels):
        block = torch.nn.Sequential(
                    torch.nn.Conv2d(kernel_size=1, in_channels=in_channels, out_channels=out_channels,stride = 1,padding = 0),
                    torch.nn.LeakyReLU(),
                )
        return block
    
    def final_block(self, in_channels, out_channels, kernel_size):
        block = torch.nn.Sequential(
                torch.nn.ConvTranspose2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, stride=2, padding=2,output_padding = 1),
                torch.nn.LeakyReLU(),
                torch.nn.Conv2d(kernel_size=1, in_channels=in_channels, out_channels=out_channels),
#                 torch.nn.Tanh()
                )
        return block
    
    def __init__(self, in_channel, out_channel):
        super(custom_AE2, self).__init__()
        #Encode
        self.conv_encode1 = self.contracting_block(in_channels=in_channel, out_channels=32,kernel_size = 5)
        self.conv_encode2 = self.contracting_block(32,64,5)
        self.conv_encode3 = self.contracting_block(64,128,5)
        self.conv_encode4 = self.contracting_block(128,128,5)
        # Bottleneck
        self.bottleneck_layer = self.bottleneck_block(128,16)
        # Decode
        self.post_bottleneck_layer = self.post_bottleneck_block(16,128)
        self.conv_decode3 = self.expansive_block(128,128,5)
        self.conv_decode2 = self.expansive_block(128,64,5)
        self.conv_decode1 = self.expansive_block(64,32,5)
        self.final_layer = self.final_block(32, out_channel,5)
    
    def forward(self, x):
        # Encode
        encode_block1 = self.conv_encode1(x)
        encode_block2 = self.conv_encode2(encode_block1)
        encode_block3 = self.conv_encode3(encode_block2)
        encode_block4 = self.conv_encode4(encode_block3)
        # Bottleneck
        bt = self.bottleneck_layer(encode_block4)
        # Decode
        cat_layer4 = self.post_bottleneck_layer(bt)
        cat_layer3 = self.conv_decode3(cat_layer4)
        cat_layer2 = self.conv_decode2(cat_layer3)
        cat_layer1 = self.conv_decode1(cat_layer2)
        final_layer = self.final_layer(cat_layer1)
        return final_layer
    
# From https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351/3
def get_gaussian_kernel(kernel_size = 5,sigma = 1.0,channels = 1):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1)/2.
    variance = sigma**2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1./(2.*np.pi*variance)) *\
                      torch.exp(
                          -torch.sum((xy_grid - mean)**2., dim=-1) /\
                          (2*variance))

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)
    
    return gaussian_kernel



class model(nn.Module):
    def __init__(self,basic_AE0,basic_AE1,basic_AE2,gauss_kernel_matrix):
        super(model,self).__init__()

        self.AE0 = basic_AE0
        self.AE1 = basic_AE1
        self.AE2 = basic_AE2

        gauss_filter = nn.Conv2d(in_channels = 1,out_channels = 1,kernel_size = 5,groups = 1,bias = False,padding = 2)
        gauss_filter.weight = nn.Parameter(gauss_kernel_matrix)
        gauss_filter.weight.requires_grad = False
        self.gauss_filter = gauss_filter

        self.upsampler = nn.Upsample(scale_factor = 2,mode = "bilinear",align_corners = True)
    
    def forward(self,I0):
        
        I1 = nn.functional.interpolate(self.gauss_filter(I0),scale_factor = 0.5,mode = "bicubic")
        I2 = nn.functional.interpolate(self.gauss_filter(I1),scale_factor = 0.5,mode = "bicubic")
        I3 = nn.functional.interpolate(self.gauss_filter(I2),scale_factor = 0.5,mode = "bicubic")

#         H2 = I2-self.upsampler(I3)
        I2_hat,bt_2 = self.AE2(I2)
#         I2_hat = H2_hat+self.upsampler(I3)

        H1 = I1-self.upsampler(I2)
        H1_hat,bt_1 = self.AE1(H1)
        I1_hat = H1_hat+self.upsampler(I2_hat)

        H0 = I0-self.upsampler(I1)
        H0_hat,bt_0 = self.AE0(H0)
        I0_hat = H0_hat+self.upsampler(I1_hat)

        return I0_hat,I1_hat,I2_hat,H0_hat,H1_hat,I0,I1,I2,H0,H1,bt_2,bt_1,bt_0

