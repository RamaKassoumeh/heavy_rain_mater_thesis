import glob
import os
import numpy as np
import h5py

from datetime import datetime, timedelta

train_data = '/home/gouda/heavyrain/RadarData_summer_21/'
satellite_data='/home/gouda/heavyrain/SatelliteData_summer_21/'
lead_time=5
# create arrays
accepted_events=[]
satellite_events=[]
threshold_persentage=10
max_threshold_persentage=0.5
max_threshold=50
zero_counter = 0
zero_persentage=20
 # chekc if the previous 6 radar files are exist
def check_previous_radar_files_exist(file_path):
    directory, filename = os.path.split(file_path)
    prefix, extension = os.path.splitext(filename)

    date_time_obj = datetime.strptime(prefix[2:12], '%y%m%d%H%M')

    lead_time_range=range(1,7)
    if lead_time==15:
        lead_time_range=range(3,9)
    if lead_time==30:
        lead_time_range=range(6,12)
    for i in lead_time_range:
        five_minutes_before = date_time_obj - timedelta(minutes=5*i)

        previous_file_name = f"{prefix[0:2]}{five_minutes_before.strftime('%y%m%d%H%M')}{extension}"

        previous_file_path = os.path.join(os.path.split(directory)[0],previous_file_name[2:8], previous_file_name)
        if not os.path.exists(previous_file_path):
            return False

    return True

def round_down_minutes(dt, round_to=5):
    # Calculate the new rounded down minute
    new_minute = (dt.minute // round_to) * round_to
    # Set the minutes to the new value and reset seconds and microseconds to zero
    dt = dt.replace(minute=new_minute, second=0, microsecond=0)
    return dt


 # chekc if the previous satellite file exists
def check_satellite_file_exist(file_path):
    directory, filename = os.path.split(file_path)
    prefix, extension = os.path.splitext(filename)

    date_time_obj = datetime.strptime(prefix[2:12], '%y%m%d%H%M')
    # round to nearest 5 minutes
    date_time_obj=round_down_minutes(date_time_obj)+ timedelta(minutes=4)
    # for sat_filename in os.listdir(satellite_data):
    # Check if the file name starts with the given prefix
    # get index of the file name that matches the prefix
    index = next((i for i, satellite_file in enumerate(satellite_list) if date_time_obj.strftime('-%Y%m%d%H%M') in satellite_file), -1)

    # if date_time_obj.strftime('-%Y%m%d%H%M') in satellite_list:
        # Return the file name that matches the prefix
        # print(f"File with prefix '{date_time_obj.strftime('-%Y%m%d%H%M')}' found: '{filename}'")
    return date_time_obj,index
    
 # chekc if the previous 6 satellite files are exist
def check_previous_satellite_files_exist(date_time_obj):

    lead_time_range=range(1,7)
    if lead_time==15:
        lead_time_range=range(3,9)
    if lead_time==30:
        lead_time_range=range(6,12)
    for i in lead_time_range:
        five_minutes_before = date_time_obj - timedelta(minutes=5*i)

        # previous_file_exists =  any(five_minutes_before.strftime('-%Y%m%d%H%M') in sat_filename for sat_filename in os.listdir(satellite_data))
        index = next((i for i, satellite_file in enumerate(satellite_list) if five_minutes_before.strftime('-%Y%m%d%H%M') in satellite_file), -1)
        if index==-1:
            return False

    return True

def check_conditions(event_persentage,event_max_precipitation,current_event_no,radar_file,total_files):
    global zero_counter
   
    date_time_obj,sat_index=check_satellite_file_exist(radar_file)
    
    if sat_index!=-1 and check_previous_radar_files_exist(radar_file) and check_previous_satellite_files_exist(date_time_obj):
        if event_persentage>=threshold_persentage or event_max_precipitation>=max_threshold_persentage:
            accepted_events.append(current_event_no)
            satellite_events.append(sat_index)
            print(f"radar {radar_file} & Satellie {satellite_list[sat_index]}")
            # if int(satellite_list[sat_index][-30:-20])-4!=int(radar_files_all[current_event_no][-14:-4]):
                # print("not matched")
            return True
        
        elif event_persentage<threshold_persentage and zero_counter/total_files*100<=zero_persentage:
            accepted_events.append(current_event_no)
            satellite_events.append(sat_index)
            zero_counter=zero_counter+1
            print(f"radar {radar_file} & Satellie {satellite_list[sat_index]}")
            # if int(satellite_list[sat_index][-30:-20])-4!=int(radar_file[-14:-4]):
                # print("not matched")
            return True
    return False


min_value=0
max_value=0

# Function to extract the date part and convert to datetime object
def extract_date(file_name):
    return file_name[-30:-20]

satellite_list=[]
radar_files_all=[]
def process_data(radar_data_folder_path):
    index=0
    total_files = 0
    for radar_folders in sorted(os.listdir(radar_data_folder_path)):
        # Construct the full path to the folder
        folder_path = os.path.join(radar_data_folder_path, radar_folders)

        # Check if the path is a directory
        if os.path.isdir(folder_path):
            radar_files = sorted(glob.glob(os.path.join(folder_path, '*.scu')))
            for radar_file in radar_files:
                # Construct the full path to the file
                radar_files_all.append(radar_file)
            total_files += len(radar_files_all)

    for sat_filename in sorted(os.listdir(satellite_data), key=extract_date):
        # Check if the file name starts with the given prefix
        if sat_filename.endswith('.tif'):
            # add the file name to the list
            satellite_list.append(sat_filename)

    global zero_counter
    global min_value
    global max_value
    zero_counter=0
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
                        index+=1
                        file.close()
                        continue
                    ds_arr = ds_arr[137:436, 58:357]
                    ds_arr = np.where(ds_arr == -999, 0, ds_arr)
                    # ds_arr = np.where(ds_arr > 100, 100, ds_arr)
                    min_value=min(np.min(ds_arr),min_value)
                    max_value=max(np.max(ds_arr),max_value)
                            
                    file.close()
                    percentage=(np.count_nonzero(ds_arr)/ ds_arr.size) * 100
                    max_precipitation_persentage=(np.count_nonzero(ds_arr>max_threshold)/ ds_arr.size) * 100
                    added_to_list=check_conditions(percentage,max_precipitation_persentage,index,radar_file,total_files)
                    if added_to_list:
                        total_sum += ds_arr.sum()
                        total_sum_square += (ds_arr ** 2).sum()
                        count+=1
                    index+=1
                    print(radar_file)
                    

    np.save(radar_data_folder_path+'/radar_data_array.npy',accepted_events)
    np.save(radar_data_folder_path+'/satellite_data_array.npy',satellite_events)
    print(f"number of accepted events: {len(accepted_events)}")
    print(f"count of all events: {index}")

    accepted_events.clear()
    satellite_events.clear()


process_data(train_data)
total_sum=0
total_sum_square=0
count=0
total_sum=0
total_sum_square=0
count=0
print(min_value)
print(max_value)
