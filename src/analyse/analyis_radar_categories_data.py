from datetime import datetime
import glob
import rasterio
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import h5py
from PIL import Image
import PIL

target_width=288
target_height=288
max_value=200
undefined_count=0
liglht_rain_count=0
moderate_rain_count=0
heavy_rain_count=0
extreme_rain_count=0
total_count=0
def read_radar_image(indx):
        global undefined_count
        global liglht_rain_count
        global moderate_rain_count
        global heavy_rain_count
        global extreme_rain_count
        global total_count
        try:
            img_path =  radar_names[indx]
            print(img_path)
            file = h5py.File(img_path, 'r')
            a_group_key = list(file.keys())[0]
            dataset_DXk = file.get(a_group_key)

            ds_arr = dataset_DXk.get('image')[:]  # the image data in an array of floats
            # the gain_rate depends on the radar file generation
            gain_rate=dataset_DXk.get('what').attrs["gain"]
            ds_arr = np.where(ds_arr >0, ds_arr * gain_rate, ds_arr)

            ds_arr=np.round(ds_arr,3)
            ds_arr = np.where(ds_arr >max_value, max_value, ds_arr)
            # Convert the 2D array to a PIL Image           
            image = Image.fromarray(ds_arr[137:436, 58:357]) # get only NRW radar area
            resized_image = image.resize((target_width, target_height),PIL.Image.NEAREST )
                    
            # Convert the resized image back to a 2D NumPy array
            ds_arr = np.array(resized_image)
            # get count of categoris
            undefined_count+=np.count_nonzero(ds_arr<0)
            liglht_rain_count+=np.count_nonzero((ds_arr>=0) & (ds_arr<2.5))
            moderate_rain_count+=np.count_nonzero((ds_arr>=2.5) & (ds_arr<10))
            heavy_rain_count+=np.count_nonzero((ds_arr>=10) & (ds_arr<50))
            extreme_rain_count+=np.count_nonzero((ds_arr>=50) & (ds_arr<=200))
            # get total count
            total_count+=ds_arr.size
            file.close()        
        except Exception as e:
            print(e)
            print(img_path)
            raise e

def read_radar_spicific_image(radar_file_path):
        global undefined_count
        global liglht_rain_count
        global moderate_rain_count
        global heavy_rain_count
        global extreme_rain_count
        global total_count
        try:
            img_path =  radar_file_path
            print(img_path)
            file = h5py.File(img_path, 'r')
            a_group_key = list(file.keys())[0]
            dataset_DXk = file.get(a_group_key)

            ds_arr = dataset_DXk.get('image')[:]  # the image data in an array of floats
            
            # print((np.count_nonzero(ds_arr)/ ds_arr.size) * 100)
            # print((np.count_nonzero(ds_arr)/ ds_arr.size) * 100)
            gain_rate=dataset_DXk.get('what').attrs["gain"]
            ds_arr = np.where(ds_arr >0, ds_arr * gain_rate, ds_arr)

            ds_arr=np.round(ds_arr,3)
            ds_arr = np.where(ds_arr >max_value, max_value, ds_arr)
            # Convert the 2D array to a PIL Image           
            image = Image.fromarray(ds_arr[137:436, 58:357]) # get only NRW radar area
            # resized_image = image.resize((128, 128))
            resized_image = image.resize((target_width, target_height),PIL.Image.NEAREST )
                    
            # Convert the resized image back to a 2D NumPy array
            ds_arr = np.array(resized_image)
            # get count of categoris
            undefined_count+=np.count_nonzero(ds_arr<0)
            liglht_rain_count+=np.count_nonzero((ds_arr>=0) & (ds_arr<2.5))
            moderate_rain_count+=np.count_nonzero((ds_arr>=2.5) & (ds_arr<7.5))
            heavy_rain_count+=np.count_nonzero((ds_arr>=7.5) & (ds_arr<50))
            extreme_rain_count+=np.count_nonzero((ds_arr>=50) & (ds_arr<=200))
            # get total count
            total_count+=ds_arr.size
            file.close()        
        except Exception as e:
            print(e)
            print(img_path)
            raise e

radar_dir='/raid/heavyrain_dataset/heavyrain/RadarData_summer_21/'
radar_data_array=np.load('/raid/heavyrain_dataset/heavyrain/RadarData_summer_21/radar_data_array.npy')
min_values =[]
max_values =[]
radar_names = []
for radar_folders in sorted(os.listdir(radar_dir)):
    # Construct the full path to the folder
    folder_path = os.path.join(radar_dir, radar_folders)

    # Check if the path is a directory
    if os.path.isdir(folder_path):
        # print(f"\nProcessing folder: {folder_path}")
        # use glob.glob to read all files in the folder order by name
        radar_files = sorted(glob.glob(os.path.join(folder_path, '*.scu')))
        radar_names.extend(radar_files)


for indx in radar_data_array:
    # if indx>10694:
    #     break
    read_radar_image(indx)

# radar_dir='/raid/heavyrain_dataset/heavyrain/RadarData_summer_21/210714/hd2107141255.scu'
# read_radar_spicific_image(radar_dir)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

with open(f'analyse_radar_test_{timestamp}.txt', 'w') as file:
    file.write(f"date: {radar_dir}\n")
    file.write(f"undefined_count: {undefined_count}, persentage is {undefined_count/total_count*100}\n")
    file.write(f"liglht_rain_count: {liglht_rain_count}, persentage is {liglht_rain_count/total_count*100}\n")
    file.write(f"moderate_rain_count: {moderate_rain_count}, persentage is {moderate_rain_count/total_count*100}\n")
    file.write(f"heavy_rain_count: {heavy_rain_count}, persentage is {heavy_rain_count/total_count*100}\n")
    file.write(f"extreme_rain_count: {extreme_rain_count}, persentage is {extreme_rain_count/total_count*100}\n")
    file.write(f"total_count: {total_count}\n")
    file.close()