import os

import glob
import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
# from torchvision.io import read_image
from PIL import Image
import PIL

class RadarFilterImageDataset(Dataset):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.img_names = []
        self.transform = transform
        self.mean=0.129
        self.std=0.857
        self.max_value=996.411
        self.min_value=-999.0 

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
        self.radar_data_array=np.load(img_dir+'/radar_data_array.npy')

    def __len__(self):

        return self.radar_data_array.__len__()

    def read_radar_image(self,indx):
        try:
            img_path =  self.img_names[indx]
            
            file = h5py.File(img_path, 'r')
            a_group_key = list(file.keys())[0]
            dataset_DXk = file.get(a_group_key)

            ds_arr = dataset_DXk.get('image')[:]  # the image data in an array of floats
            # ds_arr = np.where(ds_arr == -999, 0, ds_arr)
            # ds_arr = np.where(ds_arr >self.max_value, self.max_value, ds_arr)
            # print((np.count_nonzero(ds_arr)/ ds_arr.size) * 100)
            # print((np.count_nonzero(ds_arr)/ ds_arr.size) * 100)
            gain_rate=dataset_DXk.get('what').attrs["gain"]
            ds_arr = np.where(ds_arr >0, ds_arr * gain_rate, ds_arr)
            ds_arr=np.round(ds_arr,3)
            # ds_arr = ds_arr * gain_rate
            # Normalize data
            ds_arr=(ds_arr - self.min_value) / (self.max_value - self.min_value)
             # Convert the 2D array to a PIL Image           
            image = Image.fromarray(ds_arr)
            # resized_image = image.resize((128, 128))
            resized_image = image.resize((268, 268),PIL.Image.NEAREST )
                    
            # Convert the resized image back to a 2D NumPy array
            resized_image = np.array(resized_image)
            # resized_image=self.transform(resized_image)
            file.close()        
            return  np.array(ds_arr[268:278,268:278])
        except Exception as e:
            print(e)
            print(img_path)
            raise e

    def __getitem__(self, idx):
        resized_radar_array=[]  
        # read 6 frames as input (0.5 hours), the current is the target
        for i in range(1,7):         
            resized_image=self.read_radar_image(self.radar_data_array[idx]-i)

            resized_radar_array.append(resized_image)
                
        label_image=self.read_radar_image(self.radar_data_array[idx])

        resized_radar_array=np.stack(resized_radar_array, axis=0)
        # Add channel dim, scale pixels between 0 and 1, send to GPU
        batch = torch.tensor(resized_radar_array).unsqueeze(0)
        batch = batch.cuda()
        # batch = batch.to(self.device)
        label = torch.tensor(label_image).unsqueeze(0)
        # label=label.to(self.device)
        label=label.cuda()
        # batch = (batch - self.mean) / self.std
        # batch = batch / self.max_value
        # label=label/self.max_value
        # label = (label - self.mean) / self.std
        return batch, label
    
    