from datetime import datetime, timedelta

import PIL
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from RadarFilterRainNet3DDataset import RadarFilterRainNetDataset

from RainNet3D import RainNet
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

decimal_places = 3

# Multiply the tensor by 10^decimal_places
factor = 10 ** decimal_places
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
folder_name='radar_trainer_30M_RainNet_288_size_log_200_normalize_3d_2018'

model=RainNet()
model=torch.nn.DataParallel(model)
model.cuda()
model.load_state_dict(torch.load(folder_name+'_model.pth'), strict=False)
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


def filter_negative_values(y_true, y_pred):
    # Flatten the arrays to 1D
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    # Create a mask for values >= 0 in both actual and predicted
    mask = (y_true_flat >= 0)
    
    # Apply the mask to filter out negative values
    y_true_filtered = y_true_flat[mask]
    y_pred_filtered = y_pred_flat[mask]
    
    return y_true_filtered, y_pred_filtered

def calculate_filtered_mse(y_true, y_pred):
    y_true_filtered, y_pred_filtered = filter_negative_values(y_true, y_pred)
    mse = mean_squared_error(y_true_filtered, y_pred_filtered)
    return mse

categories_threshold={'undefined':(-999, 0),'light rain':(0, 2.5), 'moderate rain':(2.5, 7.5), 'heavy rain':(7.5, 50),'Violent rain':(50, 201)}# Function to categorize pixel values based on thresholds

# categories_threshold={'undefined':(-999, 0),'light rain':(0, 2), 'moderate rain':(2, 5), 'heavy rain':(5, 10),'Violent rain':(10, 201)}# Function to categorize pixel values based on thresholds
def categorize_pixel(value, thresholds, categories):
    for i, (lower, upper) in enumerate(thresholds):
        if lower <= value < upper:
            return categories[i]
    return categories[-1]  # Return the last category as default

# Function to calculate CSI for a single category
def calculate_cat_csi(predicted, actual, category):
    actual_label=actual.detach().cpu().numpy().astype(int)
    predicted_label=predicted.detach().cpu().numpy().astype(int)
    # Calculate confusion matrix
    cm = confusion_matrix((actual_label>= categories_threshold[category][0]).astype(int) & (actual_label< categories_threshold[category][1]).astype(int), (predicted_label>= categories_threshold[category][0]).astype(int) & (predicted_label< categories_threshold[category][1]).astype(int), labels=[0, 1])
    # Check the shape of the confusion matrix
    # Check the shape of the confusion matrix
    if cm.shape == (2, 2):
        # Unpack the confusion matrix values
        tn, fp, fn, tp = cm.ravel()

        # Calculate CSI using the formula
        if tp + fp + fn == 0:
            # Handle the case where TP + FP + FN is zero
            # csi = np.nan  
            csi = 1
        else:
            # Calculate CSI
            csi = tp / (tp + fp + fn)
    else:
        # In case the confusion matrix is not 2x2, handle appropriately
        csi = 0 
    # Calculate CSI
    return csi

def check_spetial_residual(actual_img,predicted_img):
    return (actual_img - predicted_img) ** 2
    

# Function to calculate CSI for a single category
def calculate_csi(predicted, actual, category):
    # Calculate the confusion matrix
    tn, fp, fn, tp = confusion_matrix(actual == category, predicted == category).ravel()
    # Calculate CSI
    csi = tp / (tp + fp + fn)
    return csi



def calculate_fractional_coverage(grid, lower_threshold, upper_threshold, neighborhood_size):
    """
    Calculate the fractional coverage of grid points exceeding the given threshold
    within a specified neighborhood size.

    Parameters:
    grid (np.ndarray): 2D array of precipitation values.
    lower_threshold (float): Precipitation lower threshold.
    upper_threshold (float): Precipitation upper threshold.
    neighborhood_size (int): Size of the neighborhood to consider.

    Returns:
    np.ndarray: Fractional coverage for each grid point.
    """
    grid = grid.cpu().numpy()  # Move tensor to CPU and convert to numpy array
    padded_grid = np.pad(grid, pad_width=neighborhood_size // 2, mode='constant', constant_values=0)
    # fractional_coverage = np.zeros_like(grid, dtype=float)

    # for i in range(grid.shape[2]):
    #     for j in range(grid.shape[3]):
    #         neighborhood = padded_grid[:,:,i:i + neighborhood_size, j:j + neighborhood_size]
    #         exceed_count = np.sum((neighborhood >= lower_threshold) & (neighborhood < upper_threshold))
    #         total_points = neighborhood.size
    #         fractional_coverage[i, j] = exceed_count / total_points

    fractional_coverage = np.zeros_like(grid, dtype=float)
    width=grid.shape[2]
    height=grid.shape[3]
    if len(grid.shape)==5:
        width=grid.shape[3]
        height=grid.shape[4]
    pad_size = neighborhood_size // 2
    for b in range(grid.shape[0]):
        for t in range(grid.shape[1]):
            padded_grid = np.pad(grid[b, t,0], pad_width=pad_size, mode='constant', constant_values=0)
            for i in range(width):
                for j in range(height):
                    neighborhood = padded_grid[i:i+neighborhood_size, j:j+neighborhood_size]
                    exceed_count = np.sum((neighborhood >= lower_threshold) & (neighborhood < upper_threshold))
                    total_points = neighborhood.size
                    fractional_coverage[b, t,0, i, j] = exceed_count / total_points
    

    return fractional_coverage


def calculate_fss(observed, forecasted, lower_threshold, upper_threshold, neighborhood_size):
    """
    Calculate the Fractional Skill Score (FSS) for the given observed and forecasted precipitation grids
    using a specified threshold and neighborhood size.

    Parameters:
    observed (np.ndarray): 2D array of observed precipitation values.
    forecasted (np.ndarray): 2D array of forecasted precipitation values.
    lower_threshold (float): Precipitation lower threshold.
    upper_threshold (float): Precipitation upper threshold.
    neighborhood_size (int): Size of the neighborhood to consider.

    Returns:
    float: Fractional Skill Score (FSS).
    """
    # Calculate fractional coverage for observed and forecasted grids
    observed_fractional_coverage = calculate_fractional_coverage(observed, lower_threshold, upper_threshold,
                                                                 neighborhood_size)
    forecasted_fractional_coverage = calculate_fractional_coverage(forecasted, lower_threshold, upper_threshold,
                                                                   neighborhood_size)

    # Calculate Mean Squared Error (MSE) between fractional coverages
    # mse = np.mean((observed_fractional_coverage - forecasted_fractional_coverage) ** 2)

    # # Calculate reference MSE (MSE for no-skill forecast)
    # # Assuming no-skill forecast is just the mean of the observed fractional coverage
    # reference_mse = np.mean(observed_fractional_coverage ** 2)

    # # Calculate FSS
    # if reference_mse ==0:
    #     return 0
    # fss = 1 - (mse / reference_mse)

    # Calculate the numerator and denominator for FSS
    numerator = np.sum((forecasted_fractional_coverage - observed_fractional_coverage) ** 2)
    denominator = np.sum(forecasted_fractional_coverage ** 2 + observed_fractional_coverage ** 2)
    if numerator==denominator ==0:
        return 1
    elif denominator==0:
        return 0
    # Calculate FSS
    fss = 1 - (numerator / denominator)

    return fss


# Calculate RMSE for each image
rmse_values = []

# Calculate CSI for each category across all images
csi_values = {category: [] for category in categories_threshold.keys()}

# Calculate fss for each category across all images
fss_values = {category: [] for category in categories_threshold.keys()}
output_file_path = f'results/{folder_name}_test_results_{timestamp}.txt'  # Specify the file path where you want to save the results
spatial_errors = []
neighborhood_size=3
model.eval()

with torch.no_grad():
    input, target = getitem('../RadarData_test_18/181209/hd1812090945.scu')
    output = model(input)
    actual_img=inverseTransform(target)
    predicted_img=inverseTransform(output)
    input=inverseTransform(input)
    plot_images([input[0,input.shape[1]-1],input[0,input.shape[1]-2],input[0,input.shape[1]-3],input[0,input.shape[1]-4],input[0,input.shape[1]-5],input[0,input.shape[1]-6] ,actual_img[0,0],predicted_img[0,0]], 2, 4,1,1,'test',folder_name)
        
    # Calculate RMSE
    mse = calculate_filtered_mse(actual_img.detach().cpu().numpy(), predicted_img.detach().cpu().numpy())
    rmse = np.sqrt(mse)
        

        # Flatten the images to 1D arrays for comparison
    actual_flat = actual_img.flatten()
    predicted_flat = predicted_img.flatten()
        
        # Categorize pixel values
        # actual_categorized = np.array([categorize_pixel(value, thresholds, categories) for value in actual_flat])
        # predicted_categorized = np.array([categorize_pixel(value, thresholds, categories) for value in predicted_flat])
        
        # Calculate CSI for each category
    for category in categories_threshold.keys():
        csi=calculate_cat_csi(predicted_flat, actual_flat, category)
        csi_values[category].append(csi)
        fss = calculate_fss(actual_img, predicted_img, categories_threshold[category][0],
                        categories_threshold[category][1], neighborhood_size)
        fss_values[category].append(fss)
        
print(f"Average RMSE across all images: {rmse}")
with open(output_file_path, 'w') as file:
    file.write(f"\nAverage RMSE across all images: {rmse}\n")

    # Calculate the average CSI for each category across all images
    average_csi = {category: np.nanmean(csi_values[category]) for category in categories_threshold.keys()}
    # Calculate the average FSS for each category across all images
    average_fss = {category: np.nanmean(fss_values[category]) for category in categories_threshold.keys()}

    # Display the results
    print("Average CSI for each category across all images:")
    for category, avg_csi in average_csi.items():
        print(f"{category}: {avg_csi}")
        file.write(f"\nAverage CSI for category: {category}: {avg_csi}\n")
    # Display the results
    print("Average FSS for each category across all images:")
    for category, avg_fss in average_fss.items():
        print(f"{category}: {avg_fss}")
        file.write(f"\nAverage FSS for category: {category}: {avg_fss}\n")
    file.close()