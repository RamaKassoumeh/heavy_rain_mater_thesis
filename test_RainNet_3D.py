from RadarFilterRainNet3DDataset import RadarFilterRainNetDataset

from RainNet3D import RainNet


import test_metrics

test_file_name='/raid/heavyrain_dataset/RadarData_test_summer_20/'
test_data = RadarFilterRainNetDataset(
    img_dir=test_file_name,
    transform=test_metrics.radar_transform,
    inverse_transform=test_metrics.radar_inverseTransform
)

file_name='radar_trainer_30M_RainNet_3d_Log'

model=RainNet()
test_metrics.test_phase(file_name,model,test_data,test_file_name,batch_size=200)