import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
parparent = os.path.dirname(parent)
sys.path.append(current)
sys.path.append(parent)
sys.path.append(parparent)

from dataloader.RadarFilterRainNetSatelliteDataset import RadarFilterRainNetSatelliteDataset

from models.RainNet_Satellite import RainNet
import model

train_dataset = RadarFilterRainNetSatelliteDataset(
    img_dir='/raid/heavyrain_dataset/RadarData_summer_18_19/',
    sat_dir='/raid/heavyrain_dataset/SatelliteData_summer_18_19/',
    transform=model.radar_undefined_transform,
    inverse_transform=model.radar_undefined_inverse_transform,
    sat_transform=model.satellite_transform,
    random_satellite=False
)

validate_data = RadarFilterRainNetSatelliteDataset(
    img_dir='/raid/heavyrain_dataset/RadarData_summer_20/',
    sat_dir='/raid/heavyrain_dataset/SatelliteData_summer_20/',
    transform=model.radar_undefined_transform,
    inverse_transform=model.radar_undefined_inverse_transform,
    sat_transform=model.satellite_transform,
    random_satellite=False
)

# test_data = RadarFilterRainNetSatelliteDataset(
#     img_dir='/raid/heavyrain_dataset/RadarData_summer_20/',
#     sat_dir='/raid/heavyrain_dataset/SatelliteData_summer_20/',
#     transform=model.radar_transform,
#     inverse_transform=model.radar_inverseTransform,
#     sat_transform=model.satellite_transform,
#     random_satellite=False
# )

modelRainnet=RainNet()
file_name='radar_trainer_30M_RainNet_Sat_288_size_log_200_normalize_3d_sat'

model.train_model(train_dataset,validate_data,modelRainnet,file_name,model.radar_undefined_inverse_transform,batch_size=50)