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

class custom_AE(nn.Module):
    def conv2d_block(self, in_channels, out_channels, kernel_size = 3,stride = 2,padding = 2,norm_ = True, activation_ = True):
        if norm_ and activation_:        
            block = torch.nn.Sequential(
                        torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels,stride = stride,padding = padding),
                        torch.nn.BatchNorm2d(out_channels),                        
                        torch.nn.LeakyReLU(),
                
                    )
        elif norm_:
            block = torch.nn.Sequential(
                        torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels,stride = stride,padding = padding),
                        torch.nn.BatchNorm2d(out_channels),
                    )
        elif activation_:
            block = torch.nn.Sequential(
                        torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels,stride = stride,padding = padding),
                        torch.nn.LeakyReLU(),
                    )
        else:
            block = torch.nn.Sequential(
                        torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels,stride = stride,padding = padding),

                    )
        return block
    

    def convTr2d_block(self, in_channels, out_channels, kernel_size = 3,stride = 2,padding = 1,output_padding= 1, norm_ = True, activation_ = True):
        if norm_ and activation_:        
            block = torch.nn.Sequential(
                torch.nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding,output_padding = output_padding),
                torch.nn.BatchNorm2d(out_channels),
                torch.nn.LeakyReLU(),
                )
        
        elif norm_:
            block = torch.nn.Sequential(
                torch.nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding,output_padding = output_padding),
                torch.nn.BatchNorm2d(out_channels),
                )
        elif activation_:
            block = torch.nn.Sequential(
                torch.nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding,output_padding = output_padding),
                torch.nn.LeakyReLU(),
                )
        else:
            block = torch.nn.Sequential(
                torch.nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding,output_padding = output_padding),
                )
        return block
    

    
    def __init__(self, encode_layers_info,decode_layers_info):
        super(custom_AE, self).__init__()
        '''
        Description of arguments
        
        encode_layers_info = {'layers_desc':'3 encoding layers,1 bottleneck,1 post bottleneck','layers':5,\
                      'input_channels':[1,32,64,64,16],'output_channels':[32,64,64,16,128],\
                      'kernel_size':[3,3,3,1,1],'stride': [2,2,2,1,1],'padding': [1,1,1,0,0],\
                      'norm_':[True]*3 + [False]*2,'activation_':[True]*3 + [False] + [True]
                     }

        decode_layers_info = {'layers_desc':'4 decoding layers,1 final layer','layers':4,\
                              'input_channels':[128,128,64,32],'output_channels':[128,64,32,32],\
                              'kernel_size':[3,3,3,3],'stride': [2,2,2,1],'padding': [1,1,1,1],\
                              'output_padding': [1,1,1,0],\
                              'norm_':[True]*3 + [False],'activation_':[True]*4
                             }
         '''

        #Encode
        encode_bt_layer_list = []
        for layer_ in range(encode_layers_info['layers']):
            encode_bt_layer_list.append(self.conv2d_block(encode_layers_info['input_channels'][layer_],\
                                              encode_layers_info['output_channels'][layer_], \
                                              encode_layers_info['kernel_size'][layer_],\
                                              encode_layers_info['stride'][layer_],\
                                              encode_layers_info['padding'][layer_],\
                                              encode_layers_info['norm_'][layer_], \
                                              encode_layers_info['activation_'][layer_]))
            
            
        decode_bt_layer_list = []
        for layer_ in range(decode_layers_info['layers']):
            decode_bt_layer_list.append(self.convTr2d_block(decode_layers_info['input_channels'][layer_], \
                                                          decode_layers_info['output_channels'][layer_], \
                                                          decode_layers_info['kernel_size'][layer_],\
                                                          decode_layers_info['stride'][layer_],\
                                                          decode_layers_info['padding'][layer_],\
                                                          decode_layers_info['output_padding'][layer_],\
                                                          decode_layers_info['norm_']\
                                                          [layer_], decode_layers_info['activation_'][layer_]))


        self.conv_encode1 = encode_bt_layer_list[0]
        self.conv_encode2 = encode_bt_layer_list[1]
        self.conv_encode3 = encode_bt_layer_list[2]
        self.conv_encode4 = encode_bt_layer_list[3]
        
        # Bottleneck
        self.bottleneck_layer = encode_bt_layer_list[4]
        
        # Decode
        self.post_bottleneck_layer = encode_bt_layer_list[5]
        self.conv_decode4 = decode_bt_layer_list[0]
        self.conv_decode3 = decode_bt_layer_list[1]
        self.conv_decode2 = decode_bt_layer_list[2]
        self.conv_decode1  = decode_bt_layer_list[3]
        self.final_layer  = self.conv2d_block(in_channels=32, out_channels=1, kernel_size = 1,stride = 1,\
                                              padding = 0,norm_ = False, activation_ = False)
    
    def forward(self, x):
        encode_block1 = self.conv_encode1(x)
        encode_block2 = self.conv_encode2(encode_block1)
        encode_block3 = self.conv_encode3(encode_block2)
        encode_block4 = self.conv_encode4(encode_block3)

        
        # Bottleneck
        bt = self.bottleneck_layer(encode_block4)
#         print(bt.shape)
        # Decode
        cat_layer5 = self.post_bottleneck_layer(bt)        
        cat_layer4 = self.conv_decode4(cat_layer5)
        cat_layer3 = self.conv_decode3(cat_layer4)
        cat_layer2 = self.conv_decode2(cat_layer3)
        cat_layer1 = self.conv_decode1(cat_layer2)
        final_layer = self.final_layer(cat_layer1)

        return final_layer
    
    
class custom_AE_bottom(nn.Module):
    def conv2d_block(self, in_channels, out_channels, kernel_size = 3,stride = 2,padding = 2,norm_ = True, activation_ = True):
        if norm_ and activation_:        
            block = torch.nn.Sequential(
                        torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels,stride = stride,padding = padding),
                        torch.nn.BatchNorm2d(out_channels),                        
                        torch.nn.LeakyReLU(),
                
                    )
        elif norm_:
            block = torch.nn.Sequential(
                        torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels,stride = stride,padding = padding),
                        torch.nn.BatchNorm2d(out_channels),
                    )
        elif activation_:
            block = torch.nn.Sequential(
                        torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels,stride = stride,padding = padding),
                        torch.nn.LeakyReLU(),
                    )
        else:
            block = torch.nn.Sequential(
                        torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels,stride = stride,padding = padding),

                    )
        return block
    

    def convTr2d_block(self, in_channels, out_channels, kernel_size = 3,stride = 2,padding = 1,output_padding= 1, norm_ = True, activation_ = True):
        if norm_ and activation_:        
            block = torch.nn.Sequential(
                torch.nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding,output_padding = output_padding),
                torch.nn.BatchNorm2d(out_channels),
                torch.nn.LeakyReLU(),
                )
        
        elif norm_:
            block = torch.nn.Sequential(
                torch.nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding,output_padding = output_padding),
                torch.nn.BatchNorm2d(out_channels),
                )
        elif activation_:
            block = torch.nn.Sequential(
                torch.nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding,output_padding = output_padding),
                torch.nn.LeakyReLU(),
                )
        else:
            block = torch.nn.Sequential(
                torch.nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding,output_padding = output_padding),
                )
        return block
    

    
    def __init__(self, encode_layers_info,decode_layers_info):
        super(custom_AE_bottom, self).__init__()
        '''
        Description of arguments
        
        encode_layers_info = {'layers_desc':'3 encoding layers,1 bottleneck,1 post bottleneck','layers':5,\
                      'input_channels':[1,32,64,64,16],'output_channels':[32,64,64,16,128],\
                      'kernel_size':[3,3,3,1,1],'stride': [2,2,2,1,1],'padding': [1,1,1,0,0],\
                      'norm_':[True]*3 + [False]*2,'activation_':[True]*3 + [False] + [True]
                     }

        decode_layers_info = {'layers_desc':'4 decoding layers,1 final layer','layers':4,\
                              'input_channels':[128,128,64,32],'output_channels':[128,64,32,32],\
                              'kernel_size':[3,3,3,3],'stride': [2,2,2,1],'padding': [1,1,1,1],\
                              'output_padding': [1,1,1,0],\
                              'norm_':[True]*3 + [False],'activation_':[True]*4
                             }
         '''

        #Encode
        encode_bt_layer_list = []
        for layer_ in range(encode_layers_info['layers']):
            encode_bt_layer_list.append(self.conv2d_block(encode_layers_info['input_channels'][layer_],\
                                              encode_layers_info['output_channels'][layer_], \
                                              encode_layers_info['kernel_size'][layer_],\
                                              encode_layers_info['stride'][layer_],\
                                              encode_layers_info['padding'][layer_],\
                                              encode_layers_info['norm_'][layer_], \
                                              encode_layers_info['activation_'][layer_]))
            
            
        decode_bt_layer_list = []
        for layer_ in range(decode_layers_info['layers']):
            decode_bt_layer_list.append(self.convTr2d_block(decode_layers_info['input_channels'][layer_], \
                                                          decode_layers_info['output_channels'][layer_], \
                                                          decode_layers_info['kernel_size'][layer_],\
                                                          decode_layers_info['stride'][layer_],\
                                                          decode_layers_info['padding'][layer_],\
                                                          decode_layers_info['output_padding'][layer_],\
                                                          decode_layers_info['norm_']\
                                                          [layer_], decode_layers_info['activation_'][layer_]))


        self.conv_encode1 = encode_bt_layer_list[0]
        self.conv_encode2 = encode_bt_layer_list[1]
        self.conv_encode3 = encode_bt_layer_list[2]
        
        # Bottleneck
        self.bottleneck_layer = encode_bt_layer_list[3]
        
        # Decode
        self.post_bottleneck_layer = encode_bt_layer_list[4]

        self.conv_decode3 = decode_bt_layer_list[0]
        self.conv_decode2 = decode_bt_layer_list[1]
        self.conv_decode1  = decode_bt_layer_list[2]
        self.final_layer  = self.conv2d_block(in_channels=32, out_channels=1, kernel_size = 1,stride = 1,\
                                              padding = 0,norm_ = False, activation_ = False)
    
    def forward(self, x):
        encode_block1 = self.conv_encode1(x)
        encode_block2 = self.conv_encode2(encode_block1)
        encode_block3 = self.conv_encode3(encode_block2)


        
        # Bottleneck
        bt = self.bottleneck_layer(encode_block3)
#         print(bt.shape)
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
        I2_hat = self.AE2(I2)
#         I2_hat = H2_hat+self.upsampler(I3)

        H1 = I1-self.upsampler(I2)
        H1_hat = self.AE1(H1)
        
        I1_hat = H1_hat+self.upsampler(I2_hat)

        H0 = I0-self.upsampler(I1)
        H0_hat = self.AE0(H0)
        I0_hat = H0_hat+self.upsampler(I1_hat)

        return I0_hat,I1_hat,I2_hat,H0_hat,H1_hat,I0,I1,I2,H0,H1


