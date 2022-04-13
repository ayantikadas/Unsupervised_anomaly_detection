import os
import glob
import copy
import torch
import numpy as np
from tqdm.notebook import tqdm
from torch.utils.data import Dataset
from monai.data import CacheDataset
from monai.transforms import Compose,LoadImageD,AddChannelD,SpacingD,OrientationD,ResizeD,ScaleIntensityD,ToTensorD,ScaleIntensityD,ToTensor

#### Loading torch data loader  ####
class Dataset_slice(Dataset):
    def __init__(self,slice_list):
        self.slice_list = slice_list
    def __len__(self):
        return len(self.slice_list)
    def __getitem__(self,index):
        slice_image = self.slice_list[index]
        return slice_image
    
    
#### Custom slice loader  ####
class slice_data_loader_():
    def __init__(self,datapath,start_volume,end_volume,resize_dim,resize_mode,align_corners_arg, end_slice_remove,start_slice,percentile_max,batch_size,shuffle_bool):
        self.datapath = datapath
        self.align_corners_arg = align_corners_arg
        self.filenames = []
        self.start_volume = start_volume
        self.end_volume = end_volume
        self.resize_dim = resize_dim
        self.resize_mode = resize_mode
        self.end_slice_remove = end_slice_remove
        self.start_slice = start_slice
        self.percentile_max = percentile_max
        self.min_value = []
        self.percen_value = []
        self.slice_data = []
        self.batch_size = batch_size
        self.shuffle_bool = shuffle_bool
        self.args_dict = {}
    
    
    def load_vol_data(self, filenames,resize_dim,resize_mode,align_corners_arg):    
        file_dict = []    
        for i in range(len(filenames)):
            file_dict.append({"img":filenames[i]})
        if align_corners_arg:
            transform = Compose([
                                   LoadImageD(keys="img"),
                                   AddChannelD(keys="img"),
                                   OrientationD(keys = "img",axcodes = "RAS"),
                                   ResizeD(keys = "img",spatial_size = resize_dim,mode = resize_mode ,align_corners = False),
                                ])
        else:
            transform = Compose([
                                   LoadImageD(keys="img"),
                                   AddChannelD(keys="img"),
                                   OrientationD(keys = "img",axcodes = "RAS"),
                                   ResizeD(keys = "img",spatial_size = resize_dim,mode = resize_mode),
                                ])
        
        dict_dataset =  CacheDataset(file_dict,transform)

        return dict_dataset
    def load_slices(self, train_dataset,start_slice, end_slice_remove,percentile_max):
        slice_data = []
        percen_value_list = []
        min_value_list = []
        tonorm_totensor = Compose([ToTensor()])
        for i in tqdm(range(len(train_dataset))):
            percen_value = np.percentile(train_dataset[i]["img"],percentile_max)
            min_value = np.min(train_dataset[i]["img"])
            percen_value_list.append(percen_value)
            min_value_list.append(min_value)
            no_of_slices = train_dataset[i]["img"].shape[3]    
            if (percen_value - min_value)!=0:
                img_ = (train_dataset[i]["img"][0,:,:,start_slice:-end_slice_remove] - min_value)/(percen_value - min_value)
            else:
                img_ = (train_dataset[i]["img"][0,:,:,start_slice:-end_slice_remove])
            temp = ((img_>1) + np.multiply((img_<=1),img_))
            no_of_slices = temp.shape[2]
            for j in range(no_of_slices):
                slice_data.append(torch.unsqueeze(tonorm_totensor(temp[:,:,j]),0))


        return slice_data, min_value_list, percen_value_list
    
    def get_slice_data(self):
        
        for ind_ in range(0,len(self.datapath)):
            temp_filenames = list(glob.glob(self.datapath[ind_]))[self.start_volume[ind_] : self.end_volume[ind_]]


            dict_dataset = self.load_vol_data(temp_filenames,resize_dim = self.resize_dim, resize_mode = self.resize_mode, align_corners_arg = self.align_corners_arg)

            temp_slice_data, temp_min_value, temp_percen_value = self.load_slices(dict_dataset, self.start_slice[ind_],                                                                                   self.end_slice_remove[ind_], self.percentile_max[ind_])

#             for i in range(0,len(temp_filenames)):
            self.filenames.extend(temp_filenames)
            self.min_value.extend(temp_min_value)
            self.percen_value.extend(temp_percen_value)

#             for j in range(0,len(temp_slice_data)):
            self.slice_data.extend(temp_slice_data)




        _dataset = Dataset_slice(self.slice_data)
        data_loader = torch.utils.data.DataLoader(_dataset,batch_size = self.batch_size ,shuffle = self.shuffle_bool ,num_workers= 4 )
        

        self.args_dict = {"datapath":self.datapath,"filenames":self.filenames,"start_volume":self.start_volume,"end_volume":self.end_volume, "resize_dim":self.resize_dim, "end_slice_remove": self.end_slice_remove, "start_slice": self.start_slice, "percentile_max": self.percentile_max, "min_value_vols": self.min_value, "percen_value_vols": self.percen_value, "batch_size": self.batch_size, "shuffle_bool": self.shuffle_bool }
        
        return data_loader, self.slice_data, self.args_dict
    
    