import rasterio
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import cupy as cp

# Initialize a list to hold data for each band
bands_data = {f'band_{i}': cp.array([]) for i in range(0, 11)}

def read_satellite_image(indx):
    satellite_file_name =  satellite_names[indx]
    with rasterio.open(satellite_file_name) as dataset:
        # Access the raster metadata
        data_array = dataset.read()
        if np.min(data_array)>0:
            for i in range(0, 11):
                bands_data[f'band_{i}']=cp.concatenate((bands_data[f'band_{i}'],cp.array(data_array[i].flatten())))
        else:
            print(f"negative value in {satellite_file_name}")
        # # Calculate the minimum along the first axis (axis 0)
        #     min_values.append(np.min(data_array, axis=(1,2)))
        #     # Calculate the maximum along the first axis (axis 0)
        #     max_values.append(np.max(data_array, axis=(1,2)))
        print(satellite_file_name)

sat_dir='../SatelliteData'
satellite_data_array=np.load('../RadarData_18/satellite_data_array.npy')
min_values =[]
max_values =[]
satellite_names = []
for satellite_file in sorted(os.listdir(sat_dir)):
    satellite_names.append(os.path.join(sat_dir, satellite_file))

for indx in satellite_data_array:
    # if indx>10694:
    #     break
    read_satellite_image(indx)

# min_value_all=np.minimum.reduce(min_values)
# max_value_all=np.maximum.reduce(max_values)


# with open("analyse_satellite_min_max.txt", 'w') as file:
#     file.write(f"Min values: {min_value_all}\n")
#     file.write(f"Max values: {max_value_all}\n")
#     file.close()

# Initialize dictionaries to hold the min and max values for each band
bands_min = {}
bands_max = {}
outliers_count = {}
IQR={}
# Calculate min and max for each band
for i in range(0, 11):
    band_values = bands_data[f'band_{i}']
    bands_min[f'band_{i}'] = cp.min(band_values)
    bands_max[f'band_{i}'] = cp.max(band_values)


    # Calculate Q1, Q3, and IQR
    # Q1 = pd.Series(band_values).quantile(0.25)
    Q1= cp.percentile(band_values, 25)
    # Q3 = pd.Series(band_values).quantile(0.75)
    Q3= cp.percentile(band_values, 75)
    IQR[f'band_{i}'] = Q3 - Q1
    
    # Determine outliers
    lower_bound = Q1 - 1.5 * IQR[f'band_{i}'] 
    upper_bound = Q3 + 1.5 * IQR[f'band_{i}'] 
    # outliers = [x for x in band_values if x < lower_bound or x > upper_bound]
    outliers_mark=cp.logical_or(band_values < lower_bound, band_values > upper_bound)
    outliers_count[f'band_{i}'] = cp.sum(band_values[outliers_mark])/cp.sum(band_values)*100
    print(f"finish IQR for band {i}")

# Print the min and max values for each band
# for i in range(0, 11):
#     print(f'Band {i} - Min: {bands_min[f"band_{i}"]}, Max: {bands_max[f"band_{i}"]}')


# Plot histograms for each band
plt.figure(figsize=(20, 30))
for i in range(1, 12):
    plt.subplot(11, 1, i)
    plt.hist(cp.asnumpy(bands_data[f'band_{i-1}']), bins=50, alpha=0.75)
    plt.title(f'Histogram for Band {i}')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    print(f"finish histogram for band {i}")

plt.tight_layout()
plt.savefig("output/satellite_histograms_2018.png")


with open("analyse_satellite_IQR.txt", 'w') as file:
    file.write(f"IQR values: {IQR}\n")
    file.write(f"Min values: {bands_min}\n")
    file.write(f"Max values: {bands_max}\n")
    file.write(f"outliers count persentage values: {outliers_count}\n")

    file.close()