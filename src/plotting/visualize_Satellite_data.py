
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
    transform=model.radar_undefined_transform,
    inverse_transform=model.radar_undefined_transform,
    sat_transform=model.satellite_transform,
    random_satellite=False,
    return_original=True
)


train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=1,
    shuffle=False
)
counter=0
for batch_num, (input_data, target, original_target) in enumerate(train_dataloader, 1):
    counter+=1
    inverse_target=model.radar_undefined_inverse_transform(target)
    assert torch.isclose(original_target,inverse_target).tolist(), "tensor1 and tensor2 are not equal"
    input_data=model.radar_undefined_inverse_transform(input_data)
    plot_radar_satellite_images(input_data[0,0,],batch_num,'test_satellite',
        'test_visualization',
        save_image=True)
    if counter >=100:
        break