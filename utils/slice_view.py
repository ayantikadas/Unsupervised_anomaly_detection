from ipywidgets import interact,fixed
from matplotlib import pyplot as plt
class slicer():
    
    def __init__(self,vol_3d,overlay = False):
        self.volume = vol_3d
        if overlay == True:
            self.volume1 = vol_3d[0]
            self.volume2 = vol_3d[1]
        

    def vol_slice_view(self,vol,slice_view, slice_no_x,slice_no_y,slice_no_z,fig_size_x,fig_size_y):
        plt.figure(figsize=(fig_size_x,fig_size_y))
        if slice_view == 'x':
            plt.imshow(vol[slice_no_x,:,:],cmap='gray')
            plt.colorbar(orientation='horizontal')
            plt.show()
        elif slice_view == 'y':
            plt.imshow(vol[:,slice_no_y,:],cmap='gray')
            plt.colorbar(orientation='horizontal')
            plt.show()

        elif slice_view == 'z':
            plt.imshow(vol[:,:,slice_no_z],cmap='gray')
            plt.colorbar(orientation='horizontal')
            plt.show()
            
    def vol_slice_view_overlay(self,vol1,vol2,weight,slice_view, slice_no_x,slice_no_y,slice_no_z,fig_size_x,fig_size_y):
        plt.figure(figsize=(fig_size_x,fig_size_y))
        if slice_view == 'x':
            plt.imshow((weight*vol2[slice_no_x,:,:]) + ((1-weight)*vol1[slice_no_x,:,:]),cmap='gray')
            plt.colorbar(orientation='horizontal')
            plt.show()
        elif slice_view == 'y':
            plt.imshow((weight*vol2[:,slice_no_y,:]) + ((1-weight)*vol1[:,slice_no_y,:]),cmap='gray')
            plt.colorbar(orientation='horizontal')
            plt.show()

        elif slice_view == 'z':
            plt.imshow((weight*vol2[:,:,slice_no_z]) + ((1-weight)*vol1[:,:,slice_no_z]),cmap='gray')
            plt.colorbar(orientation='horizontal')
            plt.show()
            
    def slicer_view(self):
        slice_no_x_max = self.volume.shape[0]-1
        slice_no_y_max = self.volume.shape[1]-1
        slice_no_z_max = self.volume.shape[2]-1
        return interact(self.vol_slice_view,vol=fixed(self.volume),slice_view=['x','y','z'],slice_no_x=           (0,slice_no_x_max,1),slice_no_y=(0,slice_no_y_max,1),slice_no_z=(0,slice_no_z_max,1),fig_size_x=(5,8),fig_size_y=(5,8))
    
    def slicer_view_overlay(self):
        slice_no_x_max = self.volume1.shape[0]-1
        slice_no_y_max = self.volume1.shape[1]-1
        slice_no_z_max = self.volume1.shape[2]-1
        return interact(self.vol_slice_view_overlay,vol1=fixed(self.volume1),vol2=fixed(self.volume2),weight=(0,1,0.1),slice_view=['x','y','z'],slice_no_x =           (0,slice_no_x_max,1),slice_no_y=(0,slice_no_y_max,1),slice_no_z=(0,slice_no_z_max,1),fig_size_x=(5,8),fig_size_y=(5,8))
            