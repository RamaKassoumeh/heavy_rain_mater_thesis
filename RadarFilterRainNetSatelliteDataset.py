import os

import glob
import h5py
import numpy as np
import pandas as pd
import rasterio
import torch
from torch.utils.data import Dataset
# from torchvision.io import read_image
from PIL import Image
import PIL
# from osgeo import gdal
from rasterio.enums import Resampling


class RadarFilterRainNetSatelliteDataset(Dataset):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    def __init__(self, img_dir,sat_dir, transform=None,inverse_transform=None,return_original=False,sat_transform=None):
        self.img_dir = img_dir
        self.sat_dir=sat_dir
        self.return_original=return_original
        self.img_names = []
        self.transform = transform
        self.inverse_transform=inverse_transform
        self.sat_transform=sat_transform
        self.mean=0.129
        self.std=0.857
        self.max_value=200
        # self.max_value=996.411
        # self.min_value=0
        self.min_value=-999.0 
        self.resampling_method='lanczos'
        self.target_width=288
        self.target_height=288
        # self.resampling_methods = {
        # 'near': gdal.GRA_NearestNeighbour,
        # 'bilinear': gdal.GRA_Bilinear,
        # 'cubic': gdal.GRA_Cubic,
        # 'cubicspline': gdal.GRA_CubicSpline,
        # 'lanczos': gdal.GRA_Lanczos
        # }
        # # Define options for gdal.Warp
        # self.warp_options = gdal.WarpOptions(
        #     format='MEM',
        #     width=self.target_width,
        #     height=self.target_height,
        #     resampleAlg=self.resampling_methods[self.resampling_method]
        # )

        # Walk through all directories and files
        for radar_folders in sorted(os.listdir(self.img_dir)):
            # Construct the full path to the folder
            folder_path = os.path.join(self.img_dir, radar_folders)

            # Check if the path is a directory
            if os.path.isdir(folder_path):
                # print(f"\nProcessing folder: {folder_path}")
                # use glob.glob to read all files in the folder order by name
                radar_files = sorted(glob.glob(os.path.join(folder_path, '*.scu')))
                self.img_names.extend(radar_files)
        self.satellite_names = []
        for satellite_file in sorted(os.listdir(self.sat_dir)):
            self.satellite_names.append(os.path.join(self.sat_dir, satellite_file))
        self.radar_data_array=np.load(img_dir+'/radar_data_array.npy')
        self.satellite_data_array=np.load(img_dir+'/satellite_data_array.npy')

    def __len__(self):

        return self.radar_data_array.__len__()

    def read_radar_image(self,indx,return_original=False):
        try:
            img_path =  self.img_names[indx]
            
            file = h5py.File(img_path, 'r')
            a_group_key = list(file.keys())[0]
            dataset_DXk = file.get(a_group_key)

            ds_arr = dataset_DXk.get('image')[:]  # the image data in an array of floats
            
            # print((np.count_nonzero(ds_arr)/ ds_arr.size) * 100)
            # print((np.count_nonzero(ds_arr)/ ds_arr.size) * 100)
            gain_rate=dataset_DXk.get('what').attrs["gain"]
            ds_arr = np.where(ds_arr >0, ds_arr * gain_rate, ds_arr)

            ds_arr=np.round(ds_arr,3)
            ds_arr = np.where(ds_arr >self.max_value, self.max_value, ds_arr)
            ds_arr = np.where(ds_arr <self.min_value, self.min_value, ds_arr)
            # for normalization
            # if return_original==False:
                # Normalize data
                # ds_arr=(ds_arr - self.min_value) / (self.max_value - self.min_value)
            # Convert the 2D array to a PIL Image           
            image = Image.fromarray(ds_arr[137:436, 58:357]) # get only NRW radar area
            # resized_image = image.resize((128, 128))
            resized_image = image.resize((self.target_width, self.target_height),PIL.Image.NEAREST )
                    
            # Convert the resized image back to a 2D NumPy array
            resized_image = np.array(resized_image)
            # resized_image=self.transform(resized_image)
            file.close()        
            return resized_image # ds_arr[110:366,110:366]
        except Exception as e:
            print(e)
            print(img_path)
            raise e
        
    def read_satellite_image(self,indx):
        satellite_file_name =  self.satellite_names[indx]
        with rasterio.open(satellite_file_name) as dataset:
            # Access the raster metadata
            data_array = dataset.read()
            return data_array

    def read_updample_satellite_image(self,indx):
        satellite_file_name =  self.satellite_names[indx]
        # satellite_dataset = gdal.Open(satellite_file_name)
        # # Perform the upsampling
        # output_satellite_dataset = gdal.Warp('', satellite_dataset, options=self.warp_options)
        # band_data = []
        # for i in range(1, output_satellite_dataset.RasterCount + 1):
        #     band = output_satellite_dataset.GetRasterBand(i)
        #     band_array = band.ReadAsArray()
        #     band_data.append(band_array)
        # data_array=np.stack(band_data, axis=0)
        with rasterio.open(satellite_file_name) as src:
            # Calculate the transform for the new dimensions
            transform = src.transform * src.transform.scale(
                (src.width / self.target_width),
                (src.height / self.target_height)
            )

            # Read the data from the source dataset and resample
            data_array = src.read(
                out_shape=(src.count, self.target_height, self.target_width),
                resampling=Resampling.lanczos
            )
        return data_array
        
    def __getitem__(self, idx):
        radar_array=[]  
        satellite_array=[]
        # read 6 frames as input (0.5 hours), the current is the target
        for i in range(1,7):         
            radar_image=self.read_radar_image(self.radar_data_array[idx]-i)
            radar_array.append(radar_image)
            # satellite_image=self.read_satellite_image(self.satellite_data_array[idx]-i)
            satellite_image=self.read_updample_satellite_image(self.satellite_data_array[idx]-i)
            satellite_array.append(satellite_image)

        label_image=self.read_radar_image(self.radar_data_array[idx])

        radar_array=np.stack(radar_array, axis=2)
        # satellite_array=np.stack(satellite_array, axis=3)
        # Add channel dim, scale pixels between 0 and 1, send to GPU
        # batch = torch.tensor(resized_radar_array).unsqueeze(0)
        
        # label = torch.tensor(label_image).unsqueeze(0)
        batch_radar = self.transform(radar_array)
		# add depth diminsion
        batch_radar = batch_radar.unsqueeze(1)
        # loop over satellite_array
        # Initialize an empty list to hold the tensors
        satellite_tensor_list = []
        for satellite_arr in satellite_array:
            satellite_arr = np.transpose(satellite_arr, (2, 1, 0))
            satellite_tensor = self.sat_transform(satellite_arr.astype(np.float32))
            satellite_tensor_list.append(satellite_tensor)
            # sat_tensor=torch.from_numpy(satellite_arr.astype(np.float32))
            # sat_tensors.append(sat_tensor)
        # Stack the tensors into a single tensor
        batch_satellite = torch.stack(satellite_tensor_list)    
        # batch_satellite = self.sat_transform(satellite_array)
        # batch = batch.unsqueeze(0)
        # batch = batch.unsqueeze(1)
        label=self.transform(label_image)
        label = label.unsqueeze(0)
        batch=torch.cat([batch_radar, batch_satellite], dim=1)
        # batch=batch.cuda()
        # batch_radar = batch_radar.cuda()
        # batch_satellite = batch_satellite.cuda()
        # label=label.cuda()
        if self.return_original==True:
            original_label_image=self.read_radar_image(self.radar_data_array[idx],True)
            # original_label=self.transform(original_label_image)
            original_label = torch.tensor(original_label_image).unsqueeze(0)
            original_label=original_label.cuda()
            return batch,label,original_label
        return batch, label
    
    