import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
parparent = os.path.dirname(parent)
sys.path.append(current)
sys.path.append(parent)
sys.path.append(parparent)

import ast
from datetime import datetime, timedelta

import PIL
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from dataloader.RadarFilterRainNetSatelliteDataset import RadarFilterRainNetSatelliteDataset

from models.RainNet_Satellite import RainNet
from models import model_RainNet
from plotting.plotting import plot_images

from convlstm import Seq2Seq
from torch.utils.data import DataLoader
import h5py
import os
import glob
from PIL import Image
import io
from sklearn.metrics import mean_squared_error
from rasterio.enums import Resampling

from torchvision import transforms
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

from tests.test_metrics import calculate_metrics, calculate_metrics_one_value,categories_threshold

decimal_places = 3

# Multiply the tensor by 10^decimal_places
factor = 10 ** decimal_places
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
file_name='radar_trainer_30M_RainNet_3d_Sat_summer_model_checkpoint_46'

model=RainNet()
model=torch.nn.DataParallel(model)
model.cuda()
checkpoint_path=f'{parparent}/models_file/{file_name}.pth'
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
# model.load_state_dict(torch.load(f'{parparent}/models_file/{file_name}_model.pth'), strict=False)

# from ipywidgets import widgets, HBox
radar_data_folder_path = '../RadarData_test_18/'
Satellite_dir='../SatelliteData_summer_21/'
# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)

resampling_method='lanczos'
target_width=288
target_height=288

transform=model_RainNet.radar_transform
inverseTransform=model_RainNet.radar_inverseTransform
sat_transform=model_RainNet.satellite_transform

def read_radar_image(event_path):
        try:
            img_path =  event_path
            
            file = h5py.File(img_path, 'r')
            a_group_key = list(file.keys())[0]
            dataset_DXk = file.get(a_group_key)

            ds_arr = dataset_DXk.get('image')[:]  # the image data in an array of floats
            
            # print((np.count_nonzero(ds_arr)/ ds_arr.size) * 100)
            # print((np.count_nonzero(ds_arr)/ ds_arr.size) * 100)
            gain_rate=dataset_DXk.get('what').attrs["gain"]
            ds_arr = np.where(ds_arr >0, ds_arr * gain_rate, ds_arr)

            ds_arr=np.round(ds_arr,3)
            ds_arr = np.where(ds_arr >200, 200, ds_arr)
            # Convert the 2D array to a PIL Image           
            image = Image.fromarray(ds_arr[137:436, 58:357]) # get only NRW radar area
            resized_image = image.resize((288, 288),PIL.Image.NEAREST )
                    
            # Convert the resized image back to a 2D NumPy array
            resized_image = np.array(resized_image)
            # resized_image=self.transform(resized_image)
            file.close()        
            return resized_image # ds_arr[110:366,110:366]
        except Exception as e:
            print(e)
            print(img_path)
            raise e
        
def round_down_minutes(dt, round_to=5):
    # Calculate the new rounded down minute
    new_minute = (dt.minute // round_to) * round_to
    # Set the minutes to the new value and reset seconds and microseconds to zero
    dt = dt.replace(minute=new_minute, second=0, microsecond=0)
    return dt
       
def read_updample_satellite_image(satellie_file):
        
        satellite_file_name = satellie_file
        with rasterio.open(satellite_file_name) as src:
            # Calculate the transform for the new dimensions
            transform = src.transform * src.transform.scale(
                (src.width / target_width),
                (src.height / target_height)
            )

            # Read the data from the source dataset and resample
            data_array = src.read(
                out_shape=(src.count, target_height, target_width),
                resampling=Resampling.lanczos
            )
        return data_array

def get_filename_by_prefix(directory, prefix):
    # Use glob to search for files matching the prefix pattern
    pattern = os.path.join(directory, prefix + '*')
    matching_files = glob.glob(pattern)
    
    if matching_files:
        # Return the first matching file
        return os.path.basename(matching_files[0])
    else:
        return None
def getitem(event_path):
        radar_array=[] 
        satellite_array=[]
        directory, filename = os.path.split(event_path)
        prefix, extension = os.path.splitext(filename)

        date_time_obj = datetime.strptime(prefix[2:12], '%y%m%d%H%M')
        date_time_obj_sat=round_down_minutes(date_time_obj)+ timedelta(minutes=4)
        # read 6 frames as input (0.5 hours), the current is the target
        for i in range(1,7):
            five_minutes_before = date_time_obj - timedelta(minutes=5*i)
            five_minutes_before_sat = date_time_obj_sat - timedelta(minutes=5*i)
            previous_file_name = f"{prefix[0:2]}{five_minutes_before.strftime('%y%m%d%H%M')}{extension}"

            previous_file_path = os.path.join(os.path.split(directory)[0],previous_file_name[2:8], previous_file_name) 
            resized_image=read_radar_image(previous_file_path)
            radar_array.append(resized_image)
            
            previous_Sattelie_file_name = get_filename_by_prefix(Satellite_dir,f"MSG3-SEVI-MSG15-0100-NA-{five_minutes_before_sat.strftime('%Y%m%d%H%M')}")
            satellite_image=read_updample_satellite_image(os.path.join(Satellite_dir,previous_Sattelie_file_name))
            satellite_array.append(satellite_image)

        label_image=read_radar_image(event_path)

        radar_array=np.stack(radar_array, axis=2)

        batch_radar = transform(radar_array)
		# add depth diminsion
        batch_radar = batch_radar.unsqueeze(1)
        # loop over satellite_array
        # Initialize an empty list to hold the tensors
        satellite_tensor_list = []
        for satellite_arr in satellite_array:
            satellite_arr = np.transpose(satellite_arr, (2, 1, 0))
            satellite_tensor = sat_transform(satellite_arr.astype(np.float32))
            satellite_tensor_list.append(satellite_tensor)
            # sat_tensor=torch.from_numpy(satellite_arr.astype(np.float32))
            # sat_tensors.append(sat_tensor)
        # Stack the tensors into a single tensor
        batch_satellite = torch.stack(satellite_tensor_list)    
        # batch_satellite = self.sat_transform(satellite_array)
        # batch = batch.unsqueeze(0)
        # batch = batch.unsqueeze(1)
        label=transform(label_image)
        label = label.unsqueeze(0)
        label = label.unsqueeze(0)
        batch=torch.cat([batch_radar, batch_satellite], dim=1)
        batch = batch.unsqueeze(0)
        batch=batch.cuda()
        label=label.cuda()
        return batch, label
# Define the loss function  
# test phase
# Define the rain categories and thresholds
# categories = ['undefined','light rain', 'moderate rain', 'heavy rain','violent rain']
# thresholds = [(-999, -0.1),(-0.1, 2.5), (2.5, 15), (15, 30), (30, 200)]  # Adjust based on your data

# Calculate RMSE for each image
rmse_values = []

# Calculate CSI for each category across all images
csi_values = {category: [] for category in categories_threshold.keys()}

# Calculate fss for each category across all images
fss_values = {category: [] for category in categories_threshold.keys()}
output_file_path = f'{parparent}/results/{file_name}_test_results_{timestamp}.txt'  # Specify the file path where you want to save the results

spatial_errors = []
neighborhood_size=3
model.eval()
test_file_name='../RadarData_summer_21/210714/hd2107141255.scu'
with torch.no_grad():
    input, target = getitem(test_file_name)
    output = model(input)
    actual_img=inverseTransform(target)
    predicted_img=inverseTransform(output)
    input=inverseTransform(input)
    plot_images([input[0,input.shape[1]-1],input[0,input.shape[1]-2],input[0,input.shape[1]-3],input[0,input.shape[1]-4],input[0,input.shape[1]-5],input[0,input.shape[1]-6] ,actual_img[0,0],predicted_img[0,0]], 2, 4,1,1,'test',file_name,advance_time=5)
        
    actual_flat = actual_img.flatten()
    predicted_flat = predicted_img.flatten()
    mse,csi_values,fss_values=calculate_metrics_one_value(actual_img,predicted_img)  
    rmse = np.sqrt(mse)
        
print(f"Average RMSE across all images: {rmse}")
with open(output_file_path, 'w') as file:
    file.write(f"test on event {test_file_name}\n")

    file.write(f"\nAverage RMSE across all images: {round(rmse,3)}\n")

    # Calculate the average CSI for each category across all images
    average_csi = {category: np.nanmean(csi_values[category]) for category in categories_threshold.keys()}
    # Calculate the average FSS for each category across all images
    average_fss = {category: np.nanmean(fss_values[category]) for category in categories_threshold.keys()}

    # Display the results
    print("Average CSI for each category across all images:")
    for category, avg_csi in average_csi.items():
        print(f"{category}: {round(avg_csi,3)}")
        file.write(f"\nAverage CSI for category: {category}: {round(avg_csi,3)}\n")
    # Display the results
    print("Average FSS for each category across all images:")
    for category, avg_fss in average_fss.items():
        print(f"{category}: {round(avg_fss,3)}")
        file.write(f"\nAverage FSS for category: {category}: {round(avg_fss,3)}\n")
    file.close()