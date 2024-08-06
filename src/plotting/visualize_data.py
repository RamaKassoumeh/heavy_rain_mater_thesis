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
# from RadarFilterRainNetSatelliteDataset import RadarFilterRainNetSatelliteDataset
import model_RainNet as model_RainNet
from plotting import plot_images,plot_image
from torch.utils.data import DataLoader
import numpy as np
import os
from torchvision import transforms

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

train_dataset = RadarFilterRainNetSatelliteDataset(
    img_dir='/home/gouda/heavyrain/RadarData_summer_20/',
    sat_dir='/raid/heavyrain_dataset/SatelliteData_summer_20/',
    return_original=True,
    transform=model_RainNet.radar_transform,
    inverse_transform=model_RainNet.radar_inverseTransform,
    sat_transform=model_RainNet.satellite_transform,
)

train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=1,
    shuffle=True
)

counter=0
for batch_num, (input, target, original_target) in enumerate(train_dataloader, 1):
    counter+=1
    inverse_target=model_RainNet.radar_inverseTransform(target)
    assert torch.isclose(original_target,inverse_target).tolist(), "tensor1 and tensor2 are not equal"
    input=model_RainNet.radar_inverseTransform(input)
    plot_images([input[0,input.shape[1]-1],input[0,input.shape[1]-2],input[0,input.shape[1]-1],input[0,input.shape[1]-2],input[0,input.shape[1]-3],input[0,input.shape[1]-4] ,inverse_target[0][0],original_target[0][0]],2, 4,1,batch_num,'test',"test_visualization")
    if counter >=100:
        break
