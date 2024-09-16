# Modifed RainNet: a convolutional neural network for radar & satellite-based precipitation nowcasting (Master Thesis)

## Brief description

We introduce Modified RainNet, a convolutional neural network for radar and satellite-based precipitation nowcasting. RainNet is trained to predict continuous precipitation intensities with lead times of 5, 15, and 30 minutes, using several years of quality-controlled weather radar and satellite composites. The radar data is provided by Hydro & Meteo company, and the satellite data is sourced from EUMETSAT. The data will be made available online once the HeavyRain project is completed.

## Architecture

The modified RainNet's design builds on the original RainNet (https://gmd.copernicus.org/articles/13/2631/2020/), which was designed for radar-based precipitation nowcasting. RainNet follows an encoder-decoder architecture, where the encoder progressively reduces the spatial resolution through pooling and convolutional layers, while the decoder progressively upscales the learned patterns to a higher spatial resolution using upsampling followed by convolutional layers. Skip connections from the encoder to the decoder ensure semantic connectivity between features across different layers.

In total, RainNet contains 20 3D convolutional layers, four max-pooling layers, four upsampling layers, two dropout layers, and four skip connections.
<img src="img/Rainnet 3D radar.png" alt="RainNet Radar Architecture" width="100%"/>

The radar & satellite model retains the same architecture as the original RainNet model, but the input includes both radar and satellite data, with a depth of 12 (1 for radar and 11 for satellite bands).
<img src="img/Rainnet 3D Sat.png" alt="RainNet Radar Architecture" width="100%"/>
## Basic usage

## Basic Usage

**Prerequisites**: 
* Python 3.10+
* torch 2.4.1+
* h5py 3.11+
* Pillow 9.0.1+
* rasterio 1.3.9+

We have two main models: 
* Modified RainNet model using radar data
* Extended RainNet model using radar + satellite data

### For Analyzing and Preprocessing:
- Run the file `src/analyse/analyze_satellite.py` to perform KS and KL tests on two satellite files.
- Run the file `src/analyse/analyze_satellite_data.py` to analyze the minimum and maximum values for each satellite file band.
- Run the file `src/analyse/analyze_radar_categories_data.py` to get the percentage of each category in a file or folder of files.
- For preprocessing, run the file `src/analyse/preprocess_radar_satellite_data.py`, ensuring that you change the lead_time, train_data location, and satellite_data location.
- Run `src/analyse/best_epoch.py` to select the best epoch based on the heavy rain and violent rain categories.

### For Training and Validating the Model:
- For the radar model, run `src/models/model_RainNet_3D.py`, making sure to modify `img_dir` and lead_time (to 5, 15, or 30 minutes).
- For the radar + satellite model, run `src/models/model_RainNet_Sat.py`, making sure to modify `img_dir`, `sat_dir`, and lead_time (to 5, 15, or 30 minutes).

### For Testing the Model:
- For the radar model, run `src/tests/test_RainNet_3D.py`, modifying `test_file_name` and `model_file_name` with the epoch number.
- For the radar + satellite model, run `src/tests/test_RainNet_Sat.py`, modifying `test_file_name` and `model_file_name` with the epoch number.
- To test a specific file and plot the image for the radar model, run `src/tests/test_event.py`, modifying `test_file_name`, lead_time, and `model_file_name` with the epoch number.
- To test a specific file and plot the image for the radar + satellite model, run `src/tests/test_event_satellite.py`, modifying `test_file_name`, lead_time, and `model_file_name` with the epoch number.

### For Plotting:
- All plotting scripts are located in the folder `src/plotting`.
