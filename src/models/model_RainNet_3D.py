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

import model


train_dataset = RadarFilterRainNetDataset(
    img_dir='/raid/heavyrain_dataset/RadarData_summer_18_19',
    transform=model.radar_transform,
    inverse_transform=model.radar_inverseTransform
)

validate_data = RadarFilterRainNetDataset(
    img_dir='/raid/heavyrain_dataset/RadarData_summer_20',
    transform=model.radar_transform,
    inverse_transform=model.radar_inverseTransform
)

test_data = RadarFilterRainNetDataset(
    img_dir='/raid/heavyrain_dataset/RadarData_summer_20',
    transform=model.radar_transform,
    inverse_transform=model.radar_inverseTransform
)


modelRainnet=RainNet()
file_name='radar_trainer_30M_RainNet_3d_Log_summer'
model.train_model(train_dataset,validate_data,test_data,modelRainnet,file_name,batch_size=400)