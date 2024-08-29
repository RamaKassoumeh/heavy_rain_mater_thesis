import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
parparent = os.path.dirname(parent)
sys.path.append(current)
sys.path.append(parent)
sys.path.append(parparent)


import ast
import cv2
import numpy as np
from sklearn.metrics import confusion_matrix, mean_squared_error,mean_absolute_error
import torch
from src.plotting.plotting import plot_images
from torch.utils.data import DataLoader

from datetime import datetime
from torchvision import transforms
from tqdm import tqdm

min_value=0
max_value=200

# read data from file
with open(f"{parparent}/src/analyse/analyse_satellite_IQR.txt", 'r') as file:
    lines = file.readlines()

data = {}
for line in lines:
    key, value = line.split(':', 1)
    key = key.strip()
    value = value.strip()
    value = value.replace('array', '')  # Remove 'array' to make it a valid dictionary
    data[key] = ast.literal_eval(value)

# Extracting the dictionaries
iqr_values = data.get("IQR values", {})
bands_min_values = data.get("Min values", {})
bands_max_values = data.get("Max values", {})
outliers_count_percentage_values = data.get("outliers count percentage values", {})


# def custom_transform1(x):
#     # Use PyTorch's where function to apply the transformation element-wise
#     return torch.where(x >= 0, x + 1, x)
# def custom_transform2(x):
#     # Use PyTorch's where function to apply the transformation element-wise
#     return torch.where(x < 0, 0, x)


# radar_transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Lambda(custom_transform1) ,
#     transforms.Lambda(custom_transform2) ,
#      transforms.Lambda(lambda x:  (torch.log(x+1) / torch.log(torch.tensor(max_value+2))).float()),    
# ])

# def invert_custom_transform1(x):
#     # Use PyTorch's where function to apply the transformation element-wise
#     return torch.where(x >= 0, x-1, x)
# def invert_custom_transform2(x):
#     # Use PyTorch's where function to apply the transformation element-wise
#     return torch.where(x < 0, -999, x) 
# def invert_custom_transform1(x):
#     # Use PyTorch's where function to apply the transformation element-wise
#     return torch.where(x >= 0, x-1, x)
# def invert_custom_transform2(x):
#     # Use PyTorch's where function to apply the transformation element-wise
#     return torch.where(x < 0, -999, x) 
# radar_inverseTransform= transforms.Compose([
#     transforms.Lambda(lambda x: torch.pow(max_value+2, x)-1),
#     # transforms.Lambda(invert_custom_transform2) ,
#     transforms.Lambda(invert_custom_transform1) ,
#     transforms.Lambda(invert_custom_transform2) ,
#     transforms.Lambda(lambda x: x) 
# ])

def normalize_Satellite(x):
    for i in range(x.size(0)):
        key=list(bands_min_values.keys())[i]
        x[i] = (x[i]-bands_min_values[key])/(bands_max_values[key]-bands_min_values[key])
    return x

satellite_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(normalize_Satellite),
    ])
categories_threshold={'undefined':(-999, 0),'light rain':(0, 2.5), 'moderate rain':(2.5, 7.5), 'heavy rain':(7.5, 50),'Violent rain':(50, 201)}# Function to categorize pixel values based on thresholds

def calculate_cat_csi(predicted, actual, category):
    actual_label=actual.detach().cpu().numpy().astype(float)
    predicted_label=predicted.detach().cpu().numpy().astype(float)
    # Calculate confusion matrix
    cm = confusion_matrix((actual_label>= categories_threshold[category][0]).astype(int) & (actual_label< categories_threshold[category][1]).astype(int), (predicted_label>= categories_threshold[category][0]).astype(int) & (predicted_label< categories_threshold[category][1]).astype(int), labels=[0, 1])
    # Check the shape of the confusion matrix
    if cm.shape == (2, 2):
        # Unpack the confusion matrix values
        tn, fp, fn, tp = cm.ravel()

        # Calculate CSI using the formula
        if tp + fp + fn == 0:
            # Handle the case where TP + FP + FN is zero
            csi = np.nan   
        else:
            # Calculate CSI
            csi = tp / (tp + fp + fn)
    else:
        # In case the confusion matrix is not 2x2, handle appropriately
        csi  = 0 
    # Calculate CSI
    # return csi
    return tp, fp, fn,csi

def check_spetial_residual(actual_img,predicted_img):
    return (actual_img - predicted_img) ** 2




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
    grid = grid.detach().cpu().numpy()  # Move tensor to CPU and convert to numpy array
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

def calculate_fractional_coverage_fast(grid, lower_threshold, upper_threshold, neighborhood_size):
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
    grid = grid.squeeze(1,2)
    grid = grid.detach().cpu().numpy()
    fractional_coverage = np.zeros_like(grid, dtype=float)
    for b in range(grid.shape[0]):
            BP = np.where((grid[b] >= lower_threshold) & (grid[b] < upper_threshold), 1, 0)
            # convert to float
            BP = BP.astype(float)
            # make kernel of size neighborhood_size
            kernel = np.ones((neighborhood_size, neighborhood_size))
            # apply the kernel to the BP using opencv
            fractional_coverage[b] = cv2.filter2D(BP, -1, kernel, borderType=cv2.BORDER_CONSTANT)
            # divide by the total number of points in the neighborhood
            fractional_coverage[b] = fractional_coverage[b] / neighborhood_size ** 2
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
    observed_fractional_coverage = calculate_fractional_coverage_fast(observed, lower_threshold, upper_threshold,
                                                                 neighborhood_size)
    forecasted_fractional_coverage = calculate_fractional_coverage_fast(forecasted, lower_threshold, upper_threshold,
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
       fss=np.nan
    elif denominator==0:
         fss=np.nan
    # Calculate FSS
    else:
        fss = 1 - (numerator / denominator)

    return numerator, denominator,fss

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

def filter_infinity_values(y_true, y_pred):
    # Flatten the arrays to 1D
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    # Create a mask for values >= 0 in both actual and predicted
    mask = np.logical_and(np.isfinite(y_true_flat), np.isfinite(y_pred_flat))    
    # Apply the mask to filter out negative values
    y_true_filtered = y_true_flat[mask]
    y_pred_filtered = y_pred_flat[mask]
    
    return y_true_filtered, y_pred_filtered

def calculate_filtered_mse(y_true, y_pred):
    y_true_filtered, y_pred_filtered = filter_negative_values(y_true, y_pred)
    y_true_filtered, y_pred_filtered = filter_infinity_values(y_true_filtered, y_pred_filtered)
    mse = mean_squared_error(y_true_filtered, y_pred_filtered)
    # mae= mean_absolute_error(y_true_filtered, y_pred_filtered)
    return mse

rmse_values = []
# Calculate CSI for each category across all images
csi_values = {category: [] for category in categories_threshold.keys()}

# Calculate fss for each category across all images
fss_values = {category: [] for category in categories_threshold.keys()}
tp_values = {category: [] for category in categories_threshold.keys()}
fp_values = {category: [] for category in categories_threshold.keys()}
fn_values = {category: [] for category in categories_threshold.keys()}
observed_fractional_coverage_values = {category: [] for category in categories_threshold.keys()}
forecasted_fractional_coverage_values = {category: [] for category in categories_threshold.keys()}
neighborhood_size=3

def calculate_metrics(actual_img,predicted_img):
    mse = calculate_filtered_mse(actual_img.detach().cpu().numpy(), predicted_img.detach().cpu().numpy())
    # mse=1
     # Calculate CSI for each category
    for category in categories_threshold.keys():
        # csi = calculate_csi(predicted_categorized, actual_categorized, category)
        tp_values[category],fp_values[category],fn_values[category],csi_values[category]=calculate_cat_csi(actual_img.flatten(), predicted_img.flatten(), category)
        observed_fractional_coverage_values[category], forecasted_fractional_coverage_values[category],fss_values[category]=calculate_fss(actual_img, predicted_img, categories_threshold[category][0],
                        categories_threshold[category][1], neighborhood_size)
    return mse, tp_values, fp_values, fn_values, observed_fractional_coverage_values, forecasted_fractional_coverage_values

# Initilaize RMSE for each image
def calculate_metrics_one_value(actual_img,predicted_img):
    mse = calculate_filtered_mse(actual_img.detach().cpu().numpy(), predicted_img.detach().cpu().numpy())
    # mse=1
     # Calculate CSI for each category
    for category in categories_threshold.keys():
        # csi = calculate_csi(predicted_categorized, actual_categorized, category)
        tp_values[category],fp_values[category],fn_values[category],csi_values[category]=calculate_cat_csi(actual_img.flatten(), predicted_img.flatten(), category)
        observed_fractional_coverage_values[category], forecasted_fractional_coverage_values[category],fss_values[category]=calculate_fss(actual_img, predicted_img, categories_threshold[category][0],
                        categories_threshold[category][1], neighborhood_size)

    return mse, csi_values,fss_values

csi_values_array = {category: [] for category in categories_threshold.keys()}
tp_values_array = {category: [] for category in categories_threshold.keys()}
fp_values_array = {category: [] for category in categories_threshold.keys()}
fn_values_array = {category: [] for category in categories_threshold.keys()}
# Calculate fss for each category across all images
fss_values_array = {category: [] for category in categories_threshold.keys()}
numerator_values_array = {category: [] for category in categories_threshold.keys()}
denominator_values_array = {category: [] for category in categories_threshold.keys()}

def test_phase(file_name,model,test_data,test_file_name,inverse_trans,batch_size,advance_time=5):
    test_loader = DataLoader(
            dataset=test_data,
            batch_size=batch_size,
            shuffle=False
    )
    # for batch_num, (input, target) in enumerate(tqdm(test_loader), 1):
    #     continue
    model=torch.nn.DataParallel(model)
    model.cuda()
    # model.load_state_dict(torch.load(f"{parparent}/models_file/{file_name}_model.pth"), strict=False)
    checkpoint_path=f'{parparent}/models_file/{file_name}.pth'
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    output_file_path = f'{parparent}/results/{file_name}_test_results_{timestamp}.txt'  # Specify the file path where you want to save the resultsspatial_errors = []
    model.eval()
    with torch.no_grad():
        for batch_num, (input, target) in enumerate(tqdm(test_loader), 1):
            output = model(input)
            actual_img=inverse_trans(target)
            predicted_img=inverse_trans(output)
            if batch_num%10 ==0:
                input=inverse_trans(input)
                # plot_images([input[0,0,input.shape[2]-1],input[0,0,input.shape[2]-2],input[0,0,input.shape[2]-3],input[0,0,input.shape[2]-4],input[0,0,input.shape[2]-5],input[0,0,input.shape[2]-6] ,target[0][0],output[0][0]], 2, 4,epoch,batch_num,'train',folder_name)
                plot_images([input[0,input.shape[1]-1],input[0,input.shape[1]-2],input[0,input.shape[1]-3],input[0,input.shape[1]-4],input[0,input.shape[1]-5],input[0,input.shape[1]-6] 
                             ,actual_img[0,0],predicted_img[0,0]], 2, 4,1,batch_num,'test',file_name,advance_time=advance_time)
            mse,tp_values, fp_values, fn_values,numerator,denominator=calculate_metrics(actual_img,predicted_img)
            rmse = np.sqrt(mse)
            for category in categories_threshold.keys():
                # csi_values_array[category].append(csi[category])
                tp_values_array[category].append(tp_values[category])
                fp_values_array[category].append(fp_values[category])
                fn_values_array[category].append(fn_values[category])
                # fss_values_array[category].append(fss[category])
                numerator_values_array[category].append(numerator[category])
                denominator_values_array[category].append(denominator[category])
            # # Append RMSE to list
            rmse_values.append(rmse)

    # Calculate the average RMSE across all images
    average_rmse = np.mean(rmse_values)

    # Display the results
    print(f"Average RMSE across all images: {average_rmse}")
    with open(output_file_path, 'w') as file:
        file.write(f"test on file {test_file_name}\n")
        
        file.write(f"\nAverage RMSE across all images: {round(average_rmse,3)}\n")
        # average_csi = {category: np.nanmean(csi_values_array[category]) for category in categories_threshold.keys()}
        average_csi = {category: np.sum(tp_values_array[category])/np.sum(tp_values_array[category]+fp_values_array[category]+ fn_values_array[category]) for category in categories_threshold.keys()}
        # average_fss = {category: np.nanmean(fss_values_array[category]) for category in categories_threshold.keys()}
        average_fss = {category: 1- (np.sum(numerator_values_array[category])/np.sum(denominator_values_array[category])) for category in categories_threshold.keys()}
        

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
