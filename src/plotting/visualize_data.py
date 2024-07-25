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
from RadarFilterImageDataset import RadarFilterImageDataset
from RadarFilterRainNet3DDataset import RadarFilterRainNetDataset
# from RadarFilterRainNetSatelliteDataset import RadarFilterRainNetSatelliteDataset
import model
from plotting.plotting import plot_images,plot_image
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
from torchvision import transforms

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

train_dataset = RadarFilterRainNetDataset(
    img_dir='/raid/heavyrain_dataset/RadarData_18',
    return_original=True,
    transform=model.radar_transform,
    inverse_transform=model.radar_inverseTransform
)

train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=1,
    shuffle=True
)

counter=0
for batch_num, (input, target, original_target) in enumerate(train_dataloader, 1):
    counter+=1
    target=model.radar_inverseTransform(target)
    input=model.radar_inverseTransform(input)
    plot_images([input[0,input.shape[1]-1],input[0,input.shape[1]-2],input[0,input.shape[1]-1],input[0,input.shape[1]-2],input[0,input.shape[1]-3],input[0,input.shape[1]-4] ,target[0][0],original_target[0][0]],2, 4,1,batch_num,'test',"test_visualization")
    if counter >=100:
        break
