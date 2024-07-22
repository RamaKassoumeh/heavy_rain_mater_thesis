import os

import glob
import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
# from torchvision.io import read_image
from PIL import Image

class RadarImageDataset(Dataset):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    threshold=30
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.img_names = []
        self.transform = transform

        # Walk through all directories and files
        for radar_folders in sorted(os.listdir(self.img_dir)):
            # Construct the full path to the folder
            folder_path = os.path.join(self.img_dir, radar_folders)

            # Check if the path is a directory
            if os.path.isdir(folder_path):
                print(f"\nProcessing folder: {folder_path}")
                # use glob.glob to read all files in the folder order by name
                radar_files = sorted(glob.glob(os.path.join(folder_path, '*.scu')))
                self.img_names.extend(radar_files)

    def __len__(self):
        return self.img_names.__len__()-6

    def read_radar_image(self,indx):
        try:
            img_path = os.path.join(self.img_dir, self.img_names[indx])
                
            file = h5py.File(img_path, 'r')
            a_group_key = list(file.keys())[0]
            dataset_DXk = file.get(a_group_key)

            ds_arr = dataset_DXk.get('image')[:]  # the image data in an array of floats
            ds_arr = np.where(ds_arr == -999, 0, ds_arr)
            # Convert the 2D array to a PIL Image
            image = Image.fromarray((ds_arr * 255).astype(np.uint8))
            # resized_image = image.resize((128, 128))
            resized_image = image.resize((128, 128))
                    
            # Convert the resized image back to a 2D NumPy array
            resized_image = np.array(resized_image)
            # resized_image=self.transform(resized_image)
            file.close()        
            return resized_image
        except Exception as e:
            print(e)
            print(img_path)
            raise e

    def __getitem__(self, idx):
        resized_radar_array=[]  
        if idx<len(self.img_names) - 6:
            # read 6 frames as input (0.5 hours), 7th is the target
            for i in range(0,6):         
                resized_image=self.read_radar_image(idx+i)

                resized_radar_array.append(resized_image)
                
            label_image=self.read_radar_image(idx+i)

            resized_radar_array=np.stack(resized_radar_array, axis=0)
            # Add channel dim, scale pixels between 0 and 1, send to GPU
            batch = torch.tensor(resized_radar_array).unsqueeze(0)
            batch = batch.to(self.device)
            label = torch.tensor(label_image).unsqueeze(0)
            label=label.to(self.device)
            # batch = (batch - mean) / std
            batch = batch / 136.7
            label=label/136.7
            return batch, label        
    
    