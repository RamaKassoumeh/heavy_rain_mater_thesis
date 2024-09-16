# Modifed RainNet: a convolutional neural network for radar & satellite-based precipitation nowcasting (Master Thesis)

## Brief description

Here we introduce modified RainNet, a convolutional neural network for radar and satellite-based precipitation nowcasting. RainNet was trained to predict continuous precipitation intensities at a lead time of 5, 15, and 30 minutes, using several years of quality-controlled weather radar and satellite composites. The radar data is provided by Hydro & Meteo company, and satellite data is downloaded from EUMETSAT. The data will be available online soon after the HeavyRain project is finished.

## Architecture

The modified RainNet's design was built on the [RainNet](https://gmd.copernicus.org/articles/13/2631/2020/), which was originally designed for radar-based precipitation nowcasting. RainNet follows an encoder-decoder architecture in which the encoder progressively downscales the spatial resolution using pooling, followed by convolutional layers, and the decoder progressively upscales the learned patterns to a higher spatial resolution using upsampling, followed by convolutional layers. There are skip connections from the encoder to the decoder branches in order to ensure semantic connectivity between features on different layers.

In total, RainNet has 20 3D convolutional, four max pooling, four upsampling, two dropout layers, and four skip connections.

<img src="img/Rainnet 3D radar.png" alt="RainNet Radar architecture" width="100%"/>

The radar & satellite model has the same architecture as the RainNet model, but the input size contains radar and satellite data with a depth of 12 (1 for radar and 11 for satellite bands).

<img src="img/Rainnet 3D Sat.png" alt="RainNet Radar architecture" width="100%"/>

## Basic usage

**Prerequisites**: 
* Python 3.10+, 
* torch 2.4.1+, 
* h5py 3.11+,
* Pillow 9.0.1+,
* rasterio 1.3.9+

We have two main models: 
 * Modefied RainNet model using radar data
 * Extended RainNet model using radar + satellite data


* For anlysing and preproccing:
    - run file "src/analyse/analyze_satellite.py" to do the KS and KL tests on two satellite files
    - run file "src/analyse/analyis_satellite_data.py" to analyzethe min and max of the satellite files of each band
    - run file "src/analyse/analyis_radar_categories_data.py" to get the precentage of each category in file or folder of files
    - For prepreocissing run the file "src/analyse/preprocess_radar_satellite_data.py" and make sure to chenge the lead_time, train_data location, and satellite_data location.
    - run "src/analyse/best_epoch.py" to choose the best epoch based on heavy rain and violent rain categories.

* For training and validation the model:
    - for radar model, run file src/models/model_RainNet_3D.py taking into consideration the modification of img_dir, and lead_time to (5, 15 or 30 min)
    - for radar + satellite model, run file src/models/model_RainNet_Sat.py taking into consideration the modification of img_dir, sat_dir, and lead_time to (5, 15 or 30 min)

* For testing the model:
    - for radar model, run file src/tests/test_RainNet_3D.py taking into consideration the modification of test_file_name, and model_file_name with the epoch number
    - for radar + satellite model, run file src/tests/test_RainNet_Sat.py taking into consideration the modification of test_file_name, and model_file_name with the epoch number
    - for test specific file and plot the image for radar model, run file src/tests/test_event.py taking into consideration the modification of test_file_name,lead_time, and model_file_name with the epoch number
    - for test specific file and plot the image for radar + satellite model, run file src/tests/test_event_satellite.py taking into consideration the modification of test_file_name,lead_time, and model_file_name with the epoch number

* For Plotting: all the plotting are exist in foler src/plotting
