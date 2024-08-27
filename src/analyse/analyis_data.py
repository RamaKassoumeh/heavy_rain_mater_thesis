import calendar
import glob
import os
import numpy as np
import h5py

import matplotlib.pyplot as plt
from datetime import datetime

import pandas as pd


def check_prev_frames(event,date,current_event_no):
    # for i in range(0,7):
    #     if current_event_no < 6 or percentage_array[current_event_no-i]<threshold:
    #         return
    if event>=threshold:
        events_above_threshold.append(event)
        dates_above_threshold.append(date)

def check_condition(event_persentage,date,event_max_precipitation,current_event_no):
    if event_persentage>=threshold or event_max_precipitation>=max_threshold:
        events_above_threshold.append(current_event_no)
        dates_above_threshold.append(date)
radar_data_folder_path = '../RadarData_summer_21/'

# with h5py.File("../RadarData/190622/hd1906220520.scu", 'a') as file:
#     a_group_key = list(file.keys())[0]
#     dataset_DXk = file.get(a_group_key)
#     ds_arr = dataset_DXk.get('image')[:]  # the image data in an array of floats
#     ds_arr = np.where(ds_arr == -999, 0, ds_arr)

#     percentage=(np.count_nonzero(ds_arr)/ ds_arr.size) * 100
#     # check_prev_frames(percentage,datetime_obj,len(percentage_array)-1)
#     percentage_max=(np.count_nonzero(ds_arr>100)/ ds_arr.size) * 100
#     file.close()

# create arrays
dates_array=[]
percentage_array=[]
events_above_threshold=[]
dates_above_threshold=[]

max_precipitation_array=[]
max_precipitation_array_persentage=[]
threshold=30
max_threshold=50
image_with_outliers=0
total_image_count=0
total_nan_count=0
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
            date= os.path.split(radar_file)[1][2:12]
           
            datetime_obj =  datetime.strptime(date,"%y%m%d%H%M")
            
            
            with h5py.File(radar_file, 'a') as file:
                print(radar_file)
                total_image_count+=1
                a_group_key = list(file.keys())[0]
                dataset_DXk = file.get(a_group_key)
                ds_arr = dataset_DXk.get('image')[:]  # the image data in an array of floats
                # ds_arr = np.where(ds_arr == -999, 0, ds_arr)
                gain_rate=dataset_DXk.get('what').attrs["gain"]
                ds_arr = ds_arr * gain_rate
                print((np.count_nonzero((ds_arr > 7.5))/ ds_arr.size) * 100 )
                if (np.count_nonzero((ds_arr > 50) & (ds_arr < 300))/ ds_arr.size) * 100 == 0 and (np.count_nonzero((ds_arr > 7.5))/ ds_arr.size) * 100 >3:
                    image_with_outliers+=1
                    file.close()
                    continue
                if np.max(ds_arr)>200:
                    image_with_outliers+=1
                    file.close()
                    continue
                percentage=(np.count_nonzero(ds_arr)/ ds_arr.size) * 100
                max_precipitation_array.append(np.max(ds_arr))
                max_precipitation_array_persentage.append((np.count_nonzero(ds_arr>200)/ ds_arr.size) * 100)
                
                if (np.count_nonzero(ds_arr<0)/ ds_arr.size) * 100 >36:
                    total_nan_count+=1
                # max_precipitation_array.append((np.count_nonzero(ds_arr>100)/ ds_arr.size) * 100)
                dates_array.append(datetime_obj)
                percentage_array.append(percentage)
                # check_prev_frames(percentage,datetime_obj,len(percentage_array)-1)
                check_condition(percentage,datetime_obj,np.max(ds_arr),len(percentage_array)-1)
                flat_array=ds_arr.flatten()
                file.close()
        if datetime_obj.day == calendar.monthrange(datetime_obj.year, datetime_obj.month)[1]:
            plt.clf()
            plt.figure(figsize=(12, 6))

            plt.bar(dates_array, percentage_array, label='Percentage')

            plt.axhline(y=20, color='g', linestyle='--')
            plt.axhline(y=30, color='r', linestyle='--')
            # Set labels and title
            plt.xlabel('date')
            plt.ylabel('% of non-zero values')
            plt.title(f'Radar data with Non-Zero values Persentage_{datetime_obj.year}_{datetime_obj.month}')
            plt.xticks(rotation=45)
            # Set Y-axis limits from 0 to 100
            plt.ylim(0, 100)

            # Add legend
            plt.legend()
           
           
            # # Convert dates to datetime objects
            # dates = pd.to_datetime(dates_above_threshold)

            # # Create a DataFrame with your data
            # data = pd.DataFrame({'Date': dates_above_threshold, 'Value': events_above_threshold})

            # # Generate a continuous date range
            # date_range = pd.date_range(start=min(data['Date']), end=max(data['Date']), freq='5T')

            # # Reindex your DataFrame to include missing dates
            # data = data.set_index('Date').reindex(date_range).reset_index()

            # # Plot the data
            # plt.bar(data['index'], data['Value'],color='r')
             # Show the plot
            plt.tight_layout()
            plt.savefig(f"output/persentage_{datetime_obj.year}_{datetime_obj.month}")


            print(f"persentage above threshold of {datetime_obj.year}_{datetime_obj.month} = {len(events_above_threshold)*100/len(percentage_array)}%")

            plt.clf()
            plt.figure(figsize=(12, 6))

            plt.bar(dates_array, max_precipitation_array, label='Max Precipitation')

            # plt.axhline(y=20, color='g', linestyle='--')
            # plt.axhline(y=30, color='r', linestyle='--')
            # Set labels and title
            plt.xlabel('date')
            plt.ylabel('Max Precipitation')
            plt.title(f'Max Precipitation of each Radar data {datetime_obj.year}_{datetime_obj.month}')
            plt.xticks(rotation=45)
            # Set Y-axis limits from 0 to max
            plt.ylim(0, np.max(max_precipitation_array)+10)

            # Add legend
            plt.legend()
            plt.savefig(f"output/max_precipitation_{datetime_obj.year}_{datetime_obj.month}")

            print(f"In {dates_array[np.argmax(max_precipitation_array)]} the max precipitation is {np.max(max_precipitation_array)}")
            print(f"In {dates_array[np.argmax(max_precipitation_array)]} the max precipitation persentage is {np.max(max_precipitation_array_persentage)}")

            percentage_array.clear()

            dates_array.clear()
            events_above_threshold.clear()
            dates_above_threshold.clear()
            max_precipitation_array.clear()

            max_precipitation_array_persentage.clear()

print(image_with_outliers/total_image_count*100)
print(total_nan_count/total_image_count*100)