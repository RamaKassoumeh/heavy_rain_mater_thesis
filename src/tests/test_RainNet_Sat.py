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
import models.model_RainNet as model_RainNet

radar_inverse_transform=model_RainNet.radar_inverseTransform
test_file_name='/home/gouda/heavyrain/RadarData_summer_21/'

test_data = RadarFilterRainNetSatelliteDataset(
    img_dir=test_file_name,
    sat_dir='/home/gouda/heavyrain/SatelliteData_summer_21',
    transform=model_RainNet.radar_transform,
    inverse_transform=radar_inverse_transform,
    sat_transform=model_RainNet.satellite_transform,
    random_satellite=False,
    lead_time=5
)
file_name='radar_trainer_30M_RainNet_3d_Sat_summer_model_checkpoint_46'

model=RainNet()
test_metrics.test_phase(file_name,model,test_data,test_file_name,radar_inverse_transform,batch_size=15,advance_time=5)