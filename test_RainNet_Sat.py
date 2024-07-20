from RadarFilterRainNetSatelliteDataset import RadarFilterRainNetSatelliteDataset

from RainNet_Satellite import RainNet
import test_metrics

test_file_name='/raid/heavyrain_dataset/RadarData_test_18/'

test_data = RadarFilterRainNetSatelliteDataset(
    img_dir=test_file_name,
    sat_dir='/raid/heavyrain_dataset/SatelliteData',
    transform=test_metrics.radar_transform,
    inverse_transform=test_metrics.radar_inverseTransform,
    sat_transform=test_metrics.satellite_transform,
    random_satellite=False
)
file_name='radar_trainer_30M_RainNet_Sat_288_size_log_200_normalize_3d_sat'

model=RainNet()
test_metrics.test_phase(file_name,model,test_data,test_file_name,batch_size=20)