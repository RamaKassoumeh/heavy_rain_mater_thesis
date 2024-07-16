
# Function to calculate CSI for a single category
import numpy as np
from sklearn.metrics import confusion_matrix, mean_squared_error

categories_threshold={'undefined':(-999, 0),'light rain':(0, 2.5), 'moderate rain':(2.5, 7.5), 'heavy rain':(7.5, 50),'Violent rain':(50, 201)}# Function to categorize pixel values based on thresholds

def calculate_cat_csi(predicted, actual, category):
    actual_label=actual.detach().cpu().numpy().astype(int)
    predicted_label=predicted.detach().cpu().numpy().astype(int)
    # Calculate confusion matrix
    cm = confusion_matrix((actual_label>= categories_threshold[category][0]).astype(int) & (actual_label< categories_threshold[category][1]).astype(int), (predicted_label>= categories_threshold[category][0]).astype(int) & (predicted_label< categories_threshold[category][1]).astype(int), labels=[0, 1])
    # Check the shape of the confusion matrix
    if cm.shape == (2, 2):
        # Unpack the confusion matrix values
        tn, fp, fn, tp = cm.ravel()

        # Calculate CSI using the formula
        if tp + fp + fn == 0:
            # Handle the case where TP + FP + FN is zero
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
        return 1
    # Calculate FSS
    fss = 1 - (numerator / denominator)

    return fss

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

# Calculate CSI for each category across all images
csi_values = {category: [] for category in categories_threshold.keys()}

# Calculate fss for each category across all images
fss_values = {category: [] for category in categories_threshold.keys()}
neighborhood_size=3

def calculate_metrics(actual_img,predicted_img):
    mse = calculate_filtered_mse(actual_img.detach().cpu().numpy(), predicted_img.detach().cpu().numpy())
     # Calculate CSI for each category
    for category in categories_threshold.keys():
        # csi = calculate_csi(predicted_categorized, actual_categorized, category)
        csi_values[category]=calculate_cat_csi(actual_img.flatten(), predicted_img.flatten(), category)
        fss_values[category]=calculate_fss(actual_img, predicted_img, categories_threshold[category][0],
                        categories_threshold[category][1], neighborhood_size)
    return mse,csi_values,fss_values