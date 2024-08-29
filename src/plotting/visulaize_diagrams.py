import sys
import os

import numpy as np
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
parparent = os.path.dirname(parent)
sys.path.append(current)
sys.path.append(parent)
sys.path.append(parparent)

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
# Function to extract data from event file
def extract_event_data(file_path):
    event_acc = EventAccumulator(file_path)
    event_acc.Reload()
    scalar_tags = event_acc.Tags()['scalars']
    # Extracting the scalar values
    events = event_acc.Scalars(scalar_tags[0])
    steps = [event.step for event in events]
    values = [event.value for event in events]
    values = [0 if np.isnan(value) else value for value in values]
    return steps, values

# Paths to the event files
log_training_path = f'{parparent}/runs/radar_trainer_30M_RainNet_3d_Log_20240718_013007/Training vs. Validation Loss_Training'
log_validation_path = f'{parparent}/runs/radar_trainer_30M_RainNet_3d_Log_20240718_013007/Training vs. Validation Loss_Validation'
mse_training_path = f'{parparent}/runs/radar_trainer_30M_RainNet_3d_MSE_20240718_013049/Training vs. Validation Loss_Training'
mse_validation_path = f'{parparent}/runs/radar_trainer_30M_RainNet_3d_MSE_20240718_013049/Training vs. Validation Loss_Validation'

# Extracting data from each event file
log_train_steps, log_train_values = extract_event_data(log_training_path)
log_val_steps, log_val_values = extract_event_data(log_validation_path)
mse_train_steps, mse_train_values = extract_event_data(mse_training_path)
mse_val_steps, mse_val_values = extract_event_data(mse_validation_path)

# Plotting the data
plt.figure(figsize=(10, 6))

plt.plot(log_train_steps, log_train_values, label='Logcosh Training Loss', color='blue')
plt.plot(log_val_steps, log_val_values, label='Logcosh Validation Loss', color='orange')
plt.plot(mse_train_steps, mse_train_values, label='MSE Training Loss', color='green')
plt.plot(mse_val_steps, mse_val_values, label='MSE Validation Loss', color='red')

# plt.ylim(4e-8, 0.0004)
# Use a logarithmic scale on the y-axis
plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.title('Training and Validation Loss')
plt.legend()

plt.savefig("MSE vs Log.png")

log_heavy_rain = f'{parparent}/runs/radar_trainer_30M_RainNet_3d_Log_20240718_013007/CSI values_heavy rain'
log_violent_rain = f'{parparent}/runs/radar_trainer_30M_RainNet_3d_Log_20240718_013007/CSI values_Violent rain'


mse_heavy_rain = f'{parparent}/runs/radar_trainer_30M_RainNet_3d_MSE_20240718_013049/CSI values_heavy rain'
mse_violent_rain = f'{parparent}/runs/radar_trainer_30M_RainNet_3d_MSE_20240718_013049/CSI values_Violent rain'

log_heavy_steps, log_heavy_values = extract_event_data(log_heavy_rain)
log_violent_steps, log_violent_values = extract_event_data(log_violent_rain)

mse_heavy_steps, mse_heavy_values = extract_event_data(mse_heavy_rain)
mse_violent_steps, mse_violent_values = extract_event_data(mse_violent_rain)
# Plotting the data
plt.figure(figsize=(10, 6))

plt.plot(log_heavy_steps, log_heavy_values, label='Logcosh Heavy Rain', color='orange')
plt.plot(mse_heavy_steps, mse_heavy_values, label='MSE Heavy Rain', color='red')

# plt.ylim(4e-8, 0.0004)
# Use a logarithmic scale on the y-axis
# plt.yscale('log')
plt.xlim(left=0)
plt.xlabel('Epoch')
plt.ylabel('CSI Value')
# plt.title('Training and Validation Loss')
plt.legend()

plt.savefig("MSE vs Log CSI heavyrain.png")

plt.figure(figsize=(10, 6))

plt.plot(log_violent_steps, log_violent_values, label='Logcosh Violent Rain', color='yellow')
plt.plot(mse_violent_steps, mse_violent_values, label='MSE Violent Rain', color='green')

# plt.ylim(4e-8, 0.0004)
# Use a logarithmic scale on the y-axis
# plt.yscale('log')
plt.xlim(left=0)
plt.xlabel('Epoch')
plt.ylabel('CSI Value')
# plt.title('Training and Validation Loss')
plt.legend()

plt.savefig("MSE vs Log CSI Violent.png")


log_heavy_rain = f'{parparent}/runs/radar_trainer_30M_RainNet_3d_Log_20240718_013007/FSS values_heavy rain'
log_violent_rain = f'{parparent}/runs/radar_trainer_30M_RainNet_3d_Log_20240718_013007/FSS values_Violent rain'


mse_heavy_rain = f'{parparent}/runs/radar_trainer_30M_RainNet_3d_MSE_20240718_013049/FSS values_heavy rain'
mse_violent_rain = f'{parparent}/runs/radar_trainer_30M_RainNet_3d_MSE_20240718_013049/FSS values_Violent rain'

log_heavy_steps, log_heavy_values = extract_event_data(log_heavy_rain)
log_violent_steps, log_violent_values = extract_event_data(log_violent_rain)

mse_heavy_steps, mse_heavy_values = extract_event_data(mse_heavy_rain)
mse_violent_steps, mse_violent_values = extract_event_data(mse_violent_rain)
# Plotting the data
plt.figure(figsize=(10, 6))

plt.plot(log_heavy_steps, log_heavy_values, label='Logcosh Heavy Rain', color='orange')
plt.plot(mse_heavy_steps, mse_heavy_values, label='MSE Heavy Rain', color='red')

# plt.ylim(4e-8, 0.0004)
# Use a logarithmic scale on the y-axis
# plt.yscale('log')
plt.xlim(left=0)
plt.xlabel('Epoch')
plt.ylabel('FSS Value')
# plt.title('Training and Validation Loss')
plt.legend()

plt.savefig("MSE vs Log FSS heavyrain.png")

plt.figure(figsize=(10, 6))

plt.plot(log_violent_steps, log_violent_values, label='Logcosh Violent Rain', color='yellow')
plt.plot(mse_violent_steps, mse_violent_values, label='MSE Violent Rain', color='green')

# plt.ylim(4e-8, 0.0004)
# Use a logarithmic scale on the y-axis
# plt.yscale('log')
plt.xlim(left=0)
plt.xlabel('Epoch')
plt.ylabel('FSS Value')
# plt.title('Training and Validation Loss')
plt.legend()

plt.savefig("MSE vs Log FSS Violent.png")