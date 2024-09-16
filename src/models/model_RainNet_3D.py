import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
parparent = os.path.dirname(parent)
sys.path.append(current)
sys.path.append(parent)
sys.path.append(parparent)

from dataloader.RadarFilterRainNet3DDataset import RadarFilterRainNetDataset

from models.RainNet3D import RainNet

import model_RainNet

radar_transform=model_RainNet.radar_transform
radar_inverse_transform=model_RainNet.radar_inverseTransform
lead_time=30
train_dataset = RadarFilterRainNetDataset(
    img_dir='/raid/heavyrain_dataset/RadarData_summer_18_19_min_30/',
    transform=radar_transform,
    inverse_transform=radar_inverse_transform,
    lead_time=lead_time
)

validate_data = RadarFilterRainNetDataset(
    img_dir='/raid/heavyrain_dataset/RadarData_summer_20_min_30/',
    transform=radar_transform,
    inverse_transform=radar_inverse_transform,
    lead_time=lead_time
)

# test_data = RadarFilterRainNetDataset(
#     img_dir='/home/gouda/heavyrain/RadarData_summer_20/',
#     transform=model.radar_undefined_transform,
#     inverse_transform=radar_inverse_transform
# )


modelRainnet=RainNet()
file_name='radar_trainer_30M_RainNet_3d_Log_summer_30_min'
model_RainNet.train_model(train_dataset,validate_data,modelRainnet,file_name,radar_inverse_transform,batch_size=300,advance_time=lead_time)