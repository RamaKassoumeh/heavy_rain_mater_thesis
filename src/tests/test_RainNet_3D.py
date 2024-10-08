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
import models.model_RainNet as model_RainNet

import test_metrics
radar_inverse_transform=model_RainNet.radar_inverseTransform
test_file_name='/raid/heavyrain_dataset/heavyrain/RadarData_summer_21_min_30/'
test_data = RadarFilterRainNetDataset(
    img_dir=test_file_name,
    transform=model_RainNet.radar_transform,
    inverse_transform=radar_inverse_transform,
    lead_time=30
)

model_file_name='radar_trainer_30M_RainNet_3d_Log_summer_30_min_model_checkpoint_13'

model=RainNet()
test_metrics.test_phase(model_file_name,model,test_data,test_file_name,radar_inverse_transform,batch_size=400,advance_time=30)