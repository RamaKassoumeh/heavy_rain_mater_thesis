import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
parparent = os.path.dirname(parent)
sys.path.append(current)
sys.path.append(parent)
sys.path.append(parparent)

from datetime import datetime, timedelta

import PIL
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from dataloader.RadarFilterRainNet3DDataset import RadarFilterRainNetDataset

from models.RainNet3D import RainNet
from plotting.plotting import plot_images

from convlstm import Seq2Seq
from torch.utils.data import DataLoader
import h5py
import os
import glob
from PIL import Image
import io
from sklearn.metrics import mean_squared_error


from torchvision import transforms
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

from tests.test_metrics import calculate_metrics,categories_threshold

decimal_places = 3

# Multiply the tensor by 10^decimal_places
factor = 10 ** decimal_places
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
folder_name='radar_trainer_30M_RainNet_288_size_log_200_normalize_3d_2018'

model=RainNet()
model=torch.nn.DataParallel(model)
model.cuda()
model.load_state_dict(torch.load(f'{parparent}/models_file/{folder_name}_model.pth'), strict=False)
# from ipywidgets import widgets, HBox
radar_data_folder_path = '../RadarData_test_18/'
# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)
# Load Data as Numpy Array
# MovingMNIST = np.load('mnist_test_seq.npy').transpose(1, 0, 2, 3)
min_value=0
max_value=200



def custom_transform1(x):
    # Use PyTorch's where function to apply the transformation element-wise
    return torch.where(x >= 0, x + 1, x)
def custom_transform2(x):
    # Use PyTorch's where function to apply the transformation element-wise
    return torch.where(x < 0, 0, x)


transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Lambda(lambda x: x.unsqueeze(0))  ,# Add a new dimension at position 0
    # transforms.Lambda(lambda x: x.cuda()) , # send data to cuda
    # transforms.Normalize(mean=[mean,],
    #                          std=[std,],)
    # transforms.Lambda(lambda x: (x-min_value)/(max_value-min_value)),
    # transforms.Lambda(lambda x: torch.log2(x+1))
    transforms.Lambda(custom_transform1) ,
    transforms.Lambda(custom_transform2) ,
    # transforms.Lambda(lambda x: torch.log(x+1)),
     transforms.Lambda(lambda x:  (torch.log(x+1) / torch.log(torch.tensor(max_value+1))).float()),
    # transforms.Lambda(lambda x: x.float())
    
])

# # Declaring the variable by using square kernels and equal stride
# c = nn.Conv3d(18, 35, 5, stride=1,padding="same")

# # Describing the input and output variables
# input = torch.randn(22, 18, 12, 52, 102)
# output = c(input)

# # Print output
# print(output) 


def invert_custom_transform1(x):
    # Use PyTorch's where function to apply the transformation element-wise
    return torch.where(x > -0.1, x-1, x)
def invert_custom_transform2(x):
    # Use PyTorch's where function to apply the transformation element-wise
    return torch.where(x <= -0.1, -999, x) 

inverseTransform= transforms.Compose([
    # transforms.Lambda(lambda x: x.unsqueeze(0))  ,# Add a new dimension at position 0
    # transforms.Lambda(lambda x: x.cuda()) , # send data to cuda
    # transforms.Normalize(mean=[-mean/std,],
                            #  std=[1/std,])
    # transforms.Lambda(lambda x: torch.exp(x)-1),
    transforms.Lambda(lambda x: torch.pow(max_value+1, x)-1),
    transforms.Lambda(invert_custom_transform2) ,
    transforms.Lambda(invert_custom_transform1) ,
    transforms.Lambda(invert_custom_transform2) ,
    # transforms.Lambda(lambda x: (x*(max_value - min_value))+min_value)
    transforms.Lambda(lambda x: x) 
])

test_data = RadarFilterRainNetDataset(
    img_dir='../RadarData_test_18/',
    transform=transform,
    inverse_transform=inverseTransform
)
test_loader = DataLoader(
    dataset=test_data,
    batch_size=200,
    shuffle=False
)
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
        
def getitem(event_path):
        resized_radar_array=[] 
        directory, filename = os.path.split(event_path)
        prefix, extension = os.path.splitext(filename)

        date_time_obj = datetime.strptime(prefix[2:12], '%y%m%d%H%M')
        # read 6 frames as input (0.5 hours), the current is the target
        for i in range(1, 7):
            five_minutes_before = date_time_obj - timedelta(minutes=5*i)

            previous_file_name = f"{prefix[0:2]}{five_minutes_before.strftime('%y%m%d%H%M')}{extension}"

            previous_file_path = os.path.join(os.path.split(directory)[0],previous_file_name[2:8], previous_file_name) 
            resized_image=read_radar_image(previous_file_path)
            resized_radar_array.append(resized_image)

        label_image=read_radar_image(event_path)

        resized_radar_array=np.stack(resized_radar_array, axis=2)
        
        # Add channel dim, scale pixels between 0 and 1, send to GPU
        # batch = torch.tensor(resized_radar_array).unsqueeze(0)
        
        # label = torch.tensor(label_image).unsqueeze(0)
        batch = transform(resized_radar_array)
        # add depth diminsion
        batch = batch.unsqueeze(1)
        batch = batch.unsqueeze(0)
        label=transform(label_image)
        label = label.unsqueeze(0)
        label = label.unsqueeze(0)
        batch = batch.cuda()
        label=label.cuda()
        return batch, label
# test phase
# Define the rain categories and thresholds
# categories = ['undefined','light rain', 'moderate rain', 'heavy rain','violent rain']
# thresholds = [(-999, -0.1),(-0.1, 2.5), (2.5, 15), (15, 30), (30, 200)]  # Adjust based on your data



rmse_values = []

# Calculate CSI for each category across all images
csi_values = {category: [] for category in categories_threshold.keys()}

# Calculate fss for each category across all images
fss_values = {category: [] for category in categories_threshold.keys()}
output_file_path = f'{parparent}/results/{folder_name}_test_results_{timestamp}.txt'  # Specify the file path where you want to save the results
spatial_errors = []
neighborhood_size=3
model.eval()

with torch.no_grad():
    input, target = getitem('../RadarData_19/191119/hd1911191010.scu')
    output = model(input)
    actual_img=inverseTransform(target)
    predicted_img=inverseTransform(output)
    input=inverseTransform(input)
    plot_images([input[0,input.shape[1]-1],input[0,input.shape[1]-2],input[0,input.shape[1]-3],input[0,input.shape[1]-4],input[0,input.shape[1]-5],input[0,input.shape[1]-6] ,actual_img[0,0],predicted_img[0,0]], 2, 4,1,1,'test',folder_name)
    actual_flat = actual_img.flatten()
    predicted_flat = predicted_img.flatten()
    mse,csi_values,fss_values=calculate_metrics(actual_img,predicted_img)  
    rmse = np.sqrt(mse)
        
print(f"Average RMSE across all images: {rmse}")
with open(output_file_path, 'w') as file:
    file.write(f"test on event '../RadarData_19/191119/hd1911191010.scu'\n")
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