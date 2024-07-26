import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
parparent = os.path.dirname(parent)
sys.path.append(current)
sys.path.append(parent)
sys.path.append(parparent)

from models.RainNet_Satellite import RainNet
from dataloader.RadarFilterRainNetSatelliteDataset import RadarFilterRainNetSatelliteDataset

import test_metrics

test_file_name='/raid/heavyrain_dataset/RadarData_summer_20/'

test_data = RadarFilterRainNetSatelliteDataset(
    img_dir=test_file_name,
    sat_dir='/raid/heavyrain_dataset/SatelliteData_summer_20',
    transform=test_metrics.radar_transform,
    inverse_transform=test_metrics.radar_inverseTransform,
    sat_transform=test_metrics.satellite_transform,
    random_satellite=False
)
file_name='radar_trainer_30M_RainNet_Sat_288_size_log_200_normalize_3d_sat_bigger_model'

model=RainNet()
test_metrics.test_phase(file_name,model,test_data,test_file_name,batch_size=10)