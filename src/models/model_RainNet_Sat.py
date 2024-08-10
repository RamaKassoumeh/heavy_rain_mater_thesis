import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
parparent = os.path.dirname(parent)
sys.path.append(current)
sys.path.append(parent)
sys.path.append(parparent)

from dataloader.RadarFilterRainNetSatellite15MinDataset import RadarFilterRainNetSatelliteDataset

from models.RainNet_Satellite import RainNet
import model_RainNet

radar_inverse_transform=model_RainNet.radar_inverseTransform

train_dataset = RadarFilterRainNetSatelliteDataset(
    img_dir='/raid/heavyrain_dataset/RadarData_summer_18_19_min_15/',
    sat_dir='/raid/heavyrain_dataset/SatelliteData_summer_18_19/',
    transform=model_RainNet.radar_transform,
    inverse_transform=radar_inverse_transform,
    sat_transform=model_RainNet.satellite_transform,
    random_satellite=False
)

validate_data = RadarFilterRainNetSatelliteDataset(
    img_dir='/raid/heavyrain_dataset/RadarData_summer_20_min_15/',
    sat_dir='/raid/heavyrain_dataset/SatelliteData_summer_20/',
    transform=model_RainNet.radar_transform,
    inverse_transform=radar_inverse_transform,
    sat_transform=model_RainNet.satellite_transform,
    random_satellite=False
)

# test_data = RadarFilterRainNetSatelliteDataset(
#     img_dir='/raid/heavyrain_dataset/RadarData_summer_20/',
#     sat_dir='/raid/heavyrain_dataset/SatelliteData_summer_20/',
#     transform=model.radar_transform,
#     inverse_transform=radar_inverse_transform,
#     sat_transform=model.satellite_transform,
#     random_satellite=False
# )

modelRainnet=RainNet()
file_name='radar_trainer_30M_RainNet_3d_Sat_summer_15_min'

model_RainNet.train_model(train_dataset,validate_data,modelRainnet,file_name,radar_inverse_transform,batch_size=25)