
import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
parparent = os.path.dirname(parent)
sys.path.append(current)
sys.path.append(parent)
sys.path.append(parparent)

from datetime import datetime

import torch
from dataloader.RadarFilterRainNetSatelliteDataset import RadarFilterRainNetSatelliteDataset
import models.model as model
from plotting import plot_images,plot_image,plot_radar_satellite_images
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

train_dataset = RadarFilterRainNetSatelliteDataset(
    img_dir='/home/gouda/heavyrain/RadarData_summer_18_19/',
    sat_dir='/home/gouda/heavyrain/SatelliteData_summer_18_19/',
    transform=model.radar_transform,
    inverse_transform=model.radar_inverseTransform,
    sat_transform=model.satellite_transform,
    random_satellite=False
)


train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=1,
    shuffle=False
)
counter=0
for batch_num, (input_data, target) in enumerate(train_dataloader, 1):
    counter+=1
    input_data[0,:,0]=model.radar_inverseTransform(input_data[0,:,0])
    input_data[0,:,1:12]=model.satellite_inverseTransform(input_data[0,:,1:12])    
    plot_radar_satellite_images(input_data[0,0,],batch_num,'test_satellite',
        'test_visualization',
        save_image=True)
    if counter >=100:
        break