import calendar
import glob
import os
import numpy as np
import h5py

import matplotlib.pyplot as plt
from datetime import datetime, timedelta

import pandas as pd

# create arrays
accepted_events=[]
satellite_events=[]
threshold_persentage=10
max_threshold_persentage=0.5
max_threshold=50
zero_counter = 0
zero_persentage=20

def check_previous_files_exist(file_path):
    directory, filename = os.path.split(file_path)
    prefix, extension = os.path.splitext(filename)

    date_time_obj = datetime.strptime(prefix[2:12], '%y%m%d%H%M')

    for i in range(1, 7):
        five_minutes_before = date_time_obj - timedelta(minutes=5*i)

        previous_file_name = f"{prefix[0:2]}{five_minutes_before.strftime('%y%m%d%H%M')}{extension}"

        previous_file_path = os.path.join(os.path.split(directory)[0],previous_file_name[2:8], previous_file_name)
        if not os.path.exists(previous_file_path):
            return False

    return True

def check_satellite_file_exist(file_path):
    directory, filename = os.path.split(file_path)
    prefix, extension = os.path.splitext(filename)

    date_time_obj = datetime.strptime(prefix[2:12], '%y%m%d%H%M')+ timedelta(minutes=4)
    for sat_filename in os.listdir(satellite_data):
        # Check if the file name starts with the given prefix
        if date_time_obj.strftime('-%Y%m%d%H%M') in sat_filename:
            # Return the file name that matches the prefix
            # print(f"File with prefix '{date_time_obj.strftime('-%Y%m%d%H%M')}' found: '{filename}'")
            return sat_filename
    return None

def check_previous_satellite_files_exist(file_path):
    directory, filename = os.path.split(file_path)
    prefix, extension = os.path.splitext(filename)

    # satellite file name is same the as last minutes
    date_time_obj = datetime.strptime(prefix[2:12], '%y%m%d%H%M')+ timedelta(minutes=4)

    for i in range(1, 7):
        five_minutes_before = date_time_obj - timedelta(minutes=5*i)

        previous_file_exists =  any(five_minutes_before.strftime('-%Y%m%d%H%M') in sat_filename for sat_filename in os.listdir(satellite_data))
        if not previous_file_exists:
            return False

    return True

def check_conditions(event_persentage,event_max_precipitation,current_event_no,radar_file,total_files):
    global zero_counter
    # chekc if the previous 6 radar files are exist
    satellite_file=check_satellite_file_exist(radar_file)
    if check_previous_files_exist(radar_file) and satellite_file!=None and check_previous_satellite_files_exist(radar_file):
        print(radar_file)
        if event_persentage>=threshold_persentage or event_max_precipitation>=max_threshold_persentage:
            accepted_events.append(current_event_no)
            satellite_events.append(os.path.join(satellite_data, satellite_file))
            return True
        
        elif event_persentage<threshold_persentage and zero_counter/total_files*100<=zero_persentage:
            accepted_events.append(current_event_no)
            satellite_events.append(os.path.join(satellite_data, satellite_file))
            zero_counter=zero_counter+1
    return False

train_data = '../RadarTest/'
# validate_data = '../RadarData_validate/'
# test_data = '../RadarData_test/'

satellite_data='../SatelliteData/'
min_value=0
max_value=0



def process_data(radar_data_folder_path):
    flattened_arrays = []
    index=0
    total_files = 0
    for radar_folders in sorted(os.listdir(radar_data_folder_path)):
        # Construct the full path to the folder
        folder_path = os.path.join(radar_data_folder_path, radar_folders)

        # Check if the path is a directory
        if os.path.isdir(folder_path):
            radar_files = sorted(glob.glob(os.path.join(folder_path, '*.scu')))
            total_files += len(radar_files)
    global zero_counter
    global min_value
    global max_value
    zero_counter=0
    means = []
    variances = []
    total_sum=0
    total_sum_square=0
    count=0
    for radar_folders in sorted(os.listdir(radar_data_folder_path)):
        # Construct the full path to the folder
        folder_path = os.path.join(radar_data_folder_path, radar_folders)

        # Check if the path is a directory
        if os.path.isdir(folder_path):
            # print(f"\nProcessing folder: {folder_path}")
            # use glob.glob to read all files in the folder order by name
            radar_files = sorted(glob.glob(os.path.join(folder_path, '*.scu')))
            # Walk through all directories and files inside the current folder
            radar_day_file = []
            # Process each file in the current directory
            for radar_file in radar_files:
                # Construct the full path to the file           
                with h5py.File(radar_file, 'a') as file:
                    a_group_key = list(file.keys())[0]
                    dataset_DXk = file.get(a_group_key)
                    gain_rate=dataset_DXk.get('what').attrs["gain"]
                    ds_arr = dataset_DXk.get('image')[:]  # the image data in an array of floats
                    ds_arr = np.where(ds_arr >0, ds_arr * gain_rate, ds_arr)

                    if np.max(ds_arr)>200:
                        file.close()
                        continue
                    ds_arr=ds_arr[110:360,110:390]
                    ds_arr = np.where(ds_arr == -999, 0, ds_arr)
                    # ds_arr = np.where(ds_arr > 100, 100, ds_arr)
                    min_value=min(np.min(ds_arr),min_value)
                    max_value=max(np.max(ds_arr),max_value)
                    
                    file.close()
                    percentage=(np.count_nonzero(ds_arr>0)/ ds_arr.size) * 100
                    max_precipitation=np.max(ds_arr)
                    max_precipitation_persentage=(np.count_nonzero(ds_arr>max_threshold)/ ds_arr.size) * 100
                    added_to_list=check_conditions(percentage,max_precipitation_persentage,index,radar_file,total_files)
                    if added_to_list:
                        total_sum += ds_arr.sum()
                        total_sum_square += (ds_arr ** 2).sum()
                        means.append(np.mean(ds_arr, axis=0))
                        count+=1
                    # flattened_arrays.append(ds_arr.flatten())
                    index+=1
                    print(radar_file)
                    
    # Concatenate all the flattened arrays together
    # combined_array = np.concatenate(flattened_arrays)
    print(f"the mean is {np.mean(means)}")
    # mean and std
    count=count*ds_arr.shape[0]*ds_arr.shape[1]
    total_mean = total_sum / count
    total_var  = (total_sum_square / count) - (total_mean ** 2)
    total_std  = np.sqrt(total_var)

    # output
    print('mean: '  + str(total_mean))
    print('std:  '  + str(total_std))
    # print(f"the std is {combined_array.std()}")
    # print(f"the max is {np.max(combined_array)}")
    # print(f"count of data points have error is {(np.count_nonzero(combined_array>100)/ combined_array.size) * 100}")
    np.save(radar_data_folder_path+'/radar_data_array.npy',accepted_events)
    np.save(radar_data_folder_path+'/satellite_data_array.npy',satellite_events)
    print(f"number of accepted events: {len(accepted_events)}")
    print(f"count of all events: {index}")

    accepted_events.clear()

# file=np.load(train_data+'/radar_data_array.npy')
# # check_previous_files_exist("../RadarData/230801/hd2308010020.scu")
# with h5py.File("../RadarData/230825/hd2308250320.scu", 'a') as file:
#     a_group_key = list(file.keys())[0]
#     dataset_DXk = file.get(a_group_key)
#     ds_arr = dataset_DXk.get('image')[:]  # the image data in an array of floats
#     ds_arr = np.where(ds_arr == -999, 0, ds_arr)
#     ds_arr = np.where(ds_arr > 100, 100, ds_arr)
#     file.close()
#     percentage=(np.count_nonzero(ds_arr)/ ds_arr.size) * 100
#     max_precipitation=np.max(ds_arr)
#     check_conditions(percentage,max_precipitation,1,"../RadarData/230825/hd2308250320.scu",300)

process_data(train_data)
# total_sum=0
# total_sum_square=0
# count=0
# # process_data(validate_data)
# total_sum=0
# total_sum_square=0
# count=0
# process_data(test_data)
print(min_value)
print(max_value)

# train data with cropping results
# the mean is 0.21695113269127725
# mean: 0.21695113269127766
# std:  0.9829045831795907
# number of accepted events: 37851
# count of all events: 104281

# validate data with cropping results
#the mean is 0.2173834820793326
# mean: 0.2173834820793333
# std:  0.6638427224076925
# number of accepted events: 3861
# count of all events: 8928