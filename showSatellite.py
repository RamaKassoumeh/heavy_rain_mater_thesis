

import os
import rasterio
import random
import matplotlib.pyplot as plt
from rasterio.plot import show
import shutil

destination_path = 'output/Satellite_files/'

def read_satellite_image(indx):
    satellite_file_name =  satellite_names[indx]
    with rasterio.open(satellite_file_name) as dataset:
        # Access the raster metadata
        data_array = dataset.read()
        show(dataset, cmap='viridis')
    # Adjust spacing between subplots if needed
    plt.tight_layout()
    plt.savefig(f'output/Satellite_images/'+os.path.splitext(os.path.basename(satellite_file_name))[0]+".jpg")
    shutil.copy(satellite_file_name, destination_path)


satellite_names = []
sat_dir='../SatelliteData/'
for satellite_file in sorted(os.listdir(sat_dir)):
    satellite_names.append(os.path.join(sat_dir, satellite_file))



# def action():
#     print("Enter key pressed! Performing action...")
    

for i in range (0,1000):
    random_number = random.randint(0, satellite_names.__len__()-1)
    print(i)
    read_satellite_image(random_number)
# while True:
#     user_input = input("Press Enter to perform an action (press 'q' to quit): ")
#     if user_input == "":
#         action()
#     elif user_input.lower() == "q":
#         print("Exiting...")
#         break
#     else:
#         print("Invalid input. Press Enter to perform an action or 'q' to quit.")