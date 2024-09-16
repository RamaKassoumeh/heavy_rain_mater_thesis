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

plt.plot(log_heavy_steps, log_heavy_values, label='Heavy Rain: Logcosh Model', color='orange')
plt.plot(mse_heavy_steps, mse_heavy_values, label='Heavy Rain: MSE Model', color='red')
plt.plot(log_violent_steps, log_violent_values, label='Violent Rain: Logcosh Model', color='blue')
plt.plot(mse_violent_steps, mse_violent_values, label='Violent Rain: MSE Model', color='green')
# plt.ylim(4e-8, 0.0004)
# Use a logarithmic scale on the y-axis
# plt.yscale('log')
plt.xlim(left=0)
plt.xlabel('Epoch')
plt.ylabel('CSI Value')
# plt.title('Training and Validation Loss')
plt.legend()

plt.savefig("MSE vs Log CSI.png")



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

plt.plot(log_heavy_steps, log_heavy_values, label='Heavy Rain: Logcosh Model', color='orange')
plt.plot(mse_heavy_steps, mse_heavy_values, label='Heavy Rain: MSE Model', color='red')
plt.plot(log_violent_steps, log_violent_values, label='Violent Rain: Logcosh Model', color='blue')
plt.plot(mse_violent_steps, mse_violent_values, label='Violent Rain: MSE Model', color='green')

# plt.ylim(4e-8, 0.0004)
# Use a logarithmic scale on the y-axis
# plt.yscale('log')
plt.xlim(left=0)
plt.xlabel('Epoch')
plt.ylabel('FSS Value')
# plt.title('Training and Validation Loss')
plt.legend()

plt.savefig("MSE vs Log FSS.png")

#### +5 min lead time

radar_heavy_rain = f'{parparent}/runs/radar_trainer_30M_RainNet_3d_Log_summer_20240722_161306/CSI values_heavy rain'
radar_violent_rain = f'{parparent}/runs/radar_trainer_30M_RainNet_3d_Log_summer_20240722_161306/CSI values_Violent rain'


sat_heavy_rain = f'{parparent}/runs/radar_trainer_30M_RainNet_3d_Sat_summer_20240724_233409/CSI values_heavy rain'
sat_violent_rain = f'{parparent}/runs/radar_trainer_30M_RainNet_3d_Sat_summer_20240724_233409/CSI values_Violent rain'

radar_heavy_steps, radar_heavy_values = extract_event_data(radar_heavy_rain)
radar_violent_steps, radar_violent_values = extract_event_data(radar_violent_rain)

sat_heavy_steps, sat_heavy_values = extract_event_data(sat_heavy_rain)
sat_violent_steps, sat_violent_values = extract_event_data(sat_violent_rain)

# Create a dictionary to track the last occurrence of each element
last_occurrence = {}
for i, value in enumerate(sat_heavy_steps):
    last_occurrence[value] = sat_heavy_values[i]

# Get the unique array1 and corresponding array2 based on the last occurrence
sat_heavy_steps= list(last_occurrence.keys())
sat_heavy_values = [last_occurrence[key] for key in sat_heavy_steps]

# Create a dictionary to track the last occurrence of each element
last_occurrence = {}
for i, value in enumerate(sat_violent_steps):
    last_occurrence[value] = sat_violent_values[i]

# Get the unique array1 and corresponding array2 based on the last occurrence
sat_violent_steps = list(last_occurrence.keys())
sat_violent_values = [last_occurrence[key] for key in sat_violent_steps]
# Plotting the data
plt.figure(figsize=(10, 6))

plt.plot(sat_heavy_steps, radar_heavy_values[0:50], label='Heavy Rain: Radar Model', color='orange')
plt.plot(sat_heavy_steps, sat_heavy_values[0:50], label='Heavy Rain: Radar & Satellite Model', color='red')
plt.plot(sat_violent_steps, radar_violent_values[0:50], label='Violent Rain: Radar Model', color='blue')
plt.plot(sat_violent_steps, sat_violent_values[0:50], label='Violent Rain: Radar & Satellite Model', color='green')
# plt.ylim(4e-8, 0.0004)
# Use a logarithmic scale on the y-axis
# plt.yscale('log')
plt.xlim(left=0)
plt.xlabel('Epoch')
plt.ylabel('CSI Value')
# plt.title('Training and Validation Loss')
plt.legend()

plt.savefig("Radar Model vs Radar & Satellite +5 CSI.png")
print(radar_heavy_values[34])
print(radar_violent_values[34])
print(sat_heavy_values[40])
print(sat_violent_values[40])
## FSS

radar_heavy_rain = f'{parparent}/runs/radar_trainer_30M_RainNet_3d_Log_summer_20240722_161306/FSS values_heavy rain'
radar_violent_rain = f'{parparent}/runs/radar_trainer_30M_RainNet_3d_Log_summer_20240722_161306/FSS values_Violent rain'


sat_heavy_rain = f'{parparent}/runs/radar_trainer_30M_RainNet_3d_Sat_summer_20240724_233409/FSS values_heavy rain'
sat_violent_rain = f'{parparent}/runs/radar_trainer_30M_RainNet_3d_Sat_summer_20240724_233409/FSS values_Violent rain'

radar_heavy_steps, radar_heavy_values = extract_event_data(radar_heavy_rain)
radar_violent_steps, radar_violent_values = extract_event_data(radar_violent_rain)

sat_heavy_steps, sat_heavy_values = extract_event_data(sat_heavy_rain)
sat_violent_steps, sat_violent_values = extract_event_data(sat_violent_rain)

# Create a dictionary to track the last occurrence of each element
last_occurrence = {}
for i, value in enumerate(sat_heavy_steps):
    last_occurrence[value] = sat_heavy_values[i]

# Get the unique array1 and corresponding array2 based on the last occurrence
sat_heavy_steps= list(last_occurrence.keys())
sat_heavy_values = [last_occurrence[key] for key in sat_heavy_steps]

# Create a dictionary to track the last occurrence of each element
last_occurrence = {}
for i, value in enumerate(sat_violent_steps):
    last_occurrence[value] = sat_violent_values[i]

# Get the unique array1 and corresponding array2 based on the last occurrence
sat_violent_steps = list(last_occurrence.keys())
sat_violent_values = [last_occurrence[key] for key in sat_violent_steps]
# Plotting the data
plt.figure(figsize=(10, 6))

plt.plot(sat_heavy_steps, radar_heavy_values[0:50], label='Heavy Rain: Radar Model', color='orange')
plt.plot(sat_heavy_steps, sat_heavy_values[0:50], label='Heavy Rain: Radar & Satellite Model', color='red')
plt.plot(sat_violent_steps, radar_violent_values[0:50], label='Violent Rain: Radar Model', color='blue')
plt.plot(sat_violent_steps, sat_violent_values[0:50], label='Violent Rain: Radar & Satellite Model', color='green')
# plt.ylim(4e-8, 0.0004)
# Use a logarithmic scale on the y-axis
# plt.yscale('log')
plt.xlim(left=0)
plt.xlabel('Epoch')
plt.ylabel('FSS Value')
# plt.title('Training and Validation Loss')
plt.legend()

plt.savefig("Radar Model vs Radar & Satellite +5 FSS.png")

#### +15 min lead time

radar_heavy_rain = f'{parparent}/runs/radar_trainer_30M_RainNet_3d_Log_summer_15_min_20240819_211329_2/CSI values_heavy rain'
radar_violent_rain = f'{parparent}/runs/radar_trainer_30M_RainNet_3d_Log_summer_15_min_20240819_211329_2/CSI values_Violent rain'
radar_moderate_rain = f'{parparent}/runs/radar_trainer_30M_RainNet_3d_Log_summer_15_min_20240819_211329_2/CSI values_moderate rain'
radar_light_rain = f'{parparent}/runs/radar_trainer_30M_RainNet_3d_Log_summer_15_min_20240819_211329_2/CSI values_light rain'
radar_undefined = f'{parparent}/runs/radar_trainer_30M_RainNet_3d_Log_summer_15_min_20240819_211329_2/CSI values_undefined'

sat_heavy_rain = f'{parparent}/runs/radar_trainer_30M_RainNet_3d_Sat_summer_15_min_20240819_210347_2/CSI values_heavy rain'
sat_violent_rain = f'{parparent}/runs/radar_trainer_30M_RainNet_3d_Sat_summer_15_min_20240819_210347_2/CSI values_Violent rain'
sat_moderate_rain = f'{parparent}/runs/radar_trainer_30M_RainNet_3d_Sat_summer_15_min_20240819_210347_2/CSI values_moderate rain'
sat_light_rain = f'{parparent}/runs/radar_trainer_30M_RainNet_3d_Sat_summer_15_min_20240819_210347_2/CSI values_light rain'
sat_undefined = f'{parparent}/runs/radar_trainer_30M_RainNet_3d_Sat_summer_15_min_20240819_210347_2/CSI values_undefined'

radar_heavy_steps, radar_heavy_values = extract_event_data(radar_heavy_rain)
radar_violent_steps, radar_violent_values = extract_event_data(radar_violent_rain)
radar_moderate_steps, radar_moderate_values = extract_event_data(radar_moderate_rain)   
radar_light_steps, radar_light_values = extract_event_data(radar_light_rain)
radar_undefined_steps, radar_undefined_values = extract_event_data(radar_undefined)

sat_heavy_steps, sat_heavy_values = extract_event_data(sat_heavy_rain)
sat_violent_steps, sat_violent_values = extract_event_data(sat_violent_rain)
sat_moderate_steps, sat_moderate_values = extract_event_data(sat_moderate_rain)
sat_light_steps, sat_light_values = extract_event_data(sat_light_rain)
sat_undefined_steps, sat_undefined_values = extract_event_data(sat_undefined)

# Plotting the data
plt.figure(figsize=(10, 6))

plt.plot(radar_heavy_steps, radar_heavy_values, label='Heavy Rain: Radar Model', color='orange')
plt.plot(sat_heavy_steps, sat_heavy_values, label='Heavy Rain: Radar & Satellite Model', color='red')
plt.plot(radar_violent_steps, radar_violent_values, label='Violent Rain: Radar Model', color='blue')
plt.plot(sat_violent_steps, sat_violent_values, label='Violent Rain: Radar & Satellite Model', color='green')
# plt.ylim(4e-8, 0.0004)
# Use a logarithmic scale on the y-axis
# plt.yscale('log')
plt.xlim(left=0)
plt.xlabel('Epoch')
plt.ylabel('CSI Value')
# plt.title('Training and Validation Loss')
plt.legend()

plt.savefig("Radar Model vs Radar & Satellite +15 CSI.png")

# Plotting the data
plt.figure(figsize=(10, 6))

plt.plot(radar_moderate_steps, radar_moderate_values, label='Moderate Rain: Radar Model', color='orange')
plt.plot(sat_moderate_steps, sat_moderate_values, label='Moderate Rain: Radar & Satellite Model', color='red')
plt.plot(radar_light_steps, radar_light_values, label='Light Rain: Radar Model', color='green')
plt.plot(sat_light_steps, sat_light_values, label='Light Rain: Radar & Satellite Model', color='blue')
plt.plot(radar_undefined_steps, radar_undefined_values, label='Undefined: Radar Model', color='pink')
plt.plot(sat_undefined_steps, sat_undefined_values, label='Undefined: Radar & Satellite Model', color='purple')
# plt.ylim(4e-8, 0.0004)
# Use a logarithmic scale on the y-axis
# plt.yscale('log')
plt.xlim(left=0)
plt.xlabel('Epoch')
plt.ylabel('CSI Value')
# plt.title('Training and Validation Loss')
# plt.legend()
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.savefig("Radar Model vs Radar & Satellite +15 CSI others.png", bbox_inches='tight')

##FSS
radar_heavy_rain = f'{parparent}/runs/radar_trainer_30M_RainNet_3d_Log_summer_15_min_20240819_211329_2/FSS values_heavy rain'
radar_violent_rain = f'{parparent}/runs/radar_trainer_30M_RainNet_3d_Log_summer_15_min_20240819_211329_2/FSS values_Violent rain'
radar_moderate_rain = f'{parparent}/runs/radar_trainer_30M_RainNet_3d_Log_summer_15_min_20240819_211329_2/FSS values_moderate rain'
radar_light_rain = f'{parparent}/runs/radar_trainer_30M_RainNet_3d_Log_summer_15_min_20240819_211329_2/FSS values_light rain'
radar_undefined = f'{parparent}/runs/radar_trainer_30M_RainNet_3d_Log_summer_15_min_20240819_211329_2/FSS values_undefined'

sat_heavy_rain = f'{parparent}/runs/radar_trainer_30M_RainNet_3d_Sat_summer_15_min_20240819_210347_2/FSS values_heavy rain'
sat_violent_rain = f'{parparent}/runs/radar_trainer_30M_RainNet_3d_Sat_summer_15_min_20240819_210347_2/FSS values_Violent rain'
sat_moderate_rain = f'{parparent}/runs/radar_trainer_30M_RainNet_3d_Sat_summer_15_min_20240819_210347_2/FSS values_moderate rain'
sat_light_rain = f'{parparent}/runs/radar_trainer_30M_RainNet_3d_Sat_summer_15_min_20240819_210347_2/FSS values_light rain'
sat_undefined = f'{parparent}/runs/radar_trainer_30M_RainNet_3d_Sat_summer_15_min_20240819_210347_2/FSS values_undefined'

radar_heavy_steps, radar_heavy_values = extract_event_data(radar_heavy_rain)
radar_violent_steps, radar_violent_values = extract_event_data(radar_violent_rain)
radar_moderate_steps, radar_moderate_values = extract_event_data(radar_moderate_rain)   
radar_light_steps, radar_light_values = extract_event_data(radar_light_rain)
radar_undefined_steps, radar_undefined_values = extract_event_data(radar_undefined)

sat_heavy_steps, sat_heavy_values = extract_event_data(sat_heavy_rain)
sat_violent_steps, sat_violent_values = extract_event_data(sat_violent_rain)
sat_moderate_steps, sat_moderate_values = extract_event_data(sat_moderate_rain)
sat_light_steps, sat_light_values = extract_event_data(sat_light_rain)
sat_undefined_steps, sat_undefined_values = extract_event_data(sat_undefined)

# Plotting the data
plt.figure(figsize=(10, 6))

plt.plot(radar_heavy_steps, radar_heavy_values, label='Heavy Rain: Radar Model', color='orange')
plt.plot(sat_heavy_steps, sat_heavy_values, label='Heavy Rain: Radar & Satellite Model', color='red')
plt.plot(radar_violent_steps, radar_violent_values, label='Violent Rain: Radar Model', color='blue')
plt.plot(sat_violent_steps, sat_violent_values, label='Violent Rain: Radar & Satellite Model', color='green')
# plt.ylim(4e-8, 0.0004)
# Use a logarithmic scale on the y-axis
# plt.yscale('log')
plt.xlim(left=0)
plt.xlabel('Epoch')
plt.ylabel('FSS Value')
# plt.title('Training and Validation Loss')
plt.legend()

plt.savefig("Radar Model vs Radar & Satellite +15 FSS.png")

# Plotting the data
plt.figure(figsize=(10, 6))

plt.plot(radar_moderate_steps, radar_moderate_values, label='Moderate Rain: Radar Model', color='orange')
plt.plot(sat_moderate_steps, sat_moderate_values, label='Moderate Rain: Radar & Satellite Model', color='red')
plt.plot(radar_light_steps, radar_light_values, label='Light Rain: Radar Model', color='green')
plt.plot(sat_light_steps, sat_light_values, label='Light Rain: Radar & Satellite Model', color='blue')
plt.plot(radar_undefined_steps, radar_undefined_values, label='Undefined: Radar Model', color='pink')
plt.plot(sat_undefined_steps, sat_undefined_values, label='Undefined: Radar & Satellite Model', color='purple')
# plt.ylim(4e-8, 0.0004)
# Use a logarithmic scale on the y-axis
# plt.yscale('log')
plt.xlim(left=0)
plt.xlabel('Epoch')
plt.ylabel('CSI Value')
# plt.title('Training and Validation Loss')
# plt.legend()
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.savefig("Radar Model vs Radar & Satellite +15 FSS others.png", bbox_inches='tight')

#### +30 min lead time

radar_heavy_rain = f'{parparent}/runs/radar_trainer_30M_RainNet_3d_Log_summer_30_min_20240827_131823_2/CSI values_heavy rain'
radar_violent_rain = f'{parparent}/runs/radar_trainer_30M_RainNet_3d_Log_summer_30_min_20240827_131823_2/CSI values_Violent rain'


sat_heavy_rain = f'{parparent}/runs/radar_trainer_30M_RainNet_3d_Sat_summer_30_min_20240824_052004_2/CSI values_heavy rain'
sat_violent_rain = f'{parparent}/runs/radar_trainer_30M_RainNet_3d_Sat_summer_30_min_20240824_052004_2/CSI values_Violent rain'

radar_heavy_steps, radar_heavy_values = extract_event_data(radar_heavy_rain)
radar_violent_steps, radar_violent_values = extract_event_data(radar_violent_rain)

sat_heavy_steps, sat_heavy_values = extract_event_data(sat_heavy_rain)
sat_violent_steps, sat_violent_values = extract_event_data(sat_violent_rain)

# Plotting the data
plt.figure(figsize=(10, 6))

plt.plot(radar_heavy_steps, radar_heavy_values, label='Heavy Rain: Radar Model', color='orange')
plt.plot(sat_heavy_steps, sat_heavy_values, label='Heavy Rain: Radar & Satellite Model', color='red')
plt.plot(radar_violent_steps, radar_violent_values, label='Violent Rain: Radar Model', color='blue')
plt.plot(sat_violent_steps, sat_violent_values, label='Violent Rain: Radar & Satellite Model', color='green')

# Use a logarithmic scale on the y-axis
# plt.yscale('log')
plt.xlim(left=0)
plt.xlabel('Epoch')
plt.ylabel('CSI Value')
# plt.title('Training and Validation Loss')
plt.legend()

plt.savefig("Radar Model vs Radar & Satellite +30 CSI.png")


radar_heavy_rain = f'{parparent}/runs/radar_trainer_30M_RainNet_3d_Log_summer_30_min_20240827_131823_2/FSS values_heavy rain'
radar_violent_rain = f'{parparent}/runs/radar_trainer_30M_RainNet_3d_Log_summer_30_min_20240827_131823_2/FSS values_Violent rain'


sat_heavy_rain = f'{parparent}/runs/radar_trainer_30M_RainNet_3d_Sat_summer_30_min_20240824_052004_2/FSS values_heavy rain'
sat_violent_rain = f'{parparent}/runs/radar_trainer_30M_RainNet_3d_Sat_summer_30_min_20240824_052004_2/FSS values_Violent rain'

radar_heavy_steps, radar_heavy_values = extract_event_data(radar_heavy_rain)
radar_violent_steps, radar_violent_values = extract_event_data(radar_violent_rain)

sat_heavy_steps, sat_heavy_values = extract_event_data(sat_heavy_rain)
sat_violent_steps, sat_violent_values = extract_event_data(sat_violent_rain)

# Plotting the data
plt.figure(figsize=(10, 6))

plt.plot(radar_heavy_steps, radar_heavy_values, label='Heavy Rain: Radar Model', color='orange')
plt.plot(sat_heavy_steps, sat_heavy_values, label='Heavy Rain: Radar & Satellite Model', color='red')
plt.plot(radar_violent_steps, radar_violent_values, label='Violent Rain: Radar Model', color='blue')
plt.plot(sat_violent_steps, sat_violent_values, label='Violent Rain: Radar & Satellite Model', color='green')

# Use a logarithmic scale on the y-axis
# plt.yscale('log')
plt.xlim(left=0)
plt.xlabel('Epoch')
plt.ylabel('FSS Value')
# plt.title('Training and Validation Loss')
plt.legend()

plt.savefig("Radar Model vs Radar & Satellite +30 FSS.png")




# Define the heavyrain values with lead time
x = [5, 15, 30]
radar = [0.672, 0.368, 0.14]
satellite = [0.707, 0.379, 0.141]

# Create the plot
plt.figure(figsize=(8, 6))
plt.plot(x, radar, marker='o', linestyle='-', color='b', label='Radar Model')
plt.plot(x, satellite, marker='o', linestyle='--', color='g', label='Radar & Satellite Model')

# Add titles and labels
plt.title("Heavy rain")
plt.xlabel("Lead Time (min)")
plt.ylabel("CSI Value")
plt.legend()  # Show the legend
plt.grid(True)
# Adjust layout and save the plot
plt.tight_layout()
plt.savefig('heavyrian lead time.png')


# Define the heavyrain values with lead time
x = [5, 15, 30]
radar = [0.417, 0.03, 0.0]
satellite = [0.452, 0.039, 0.003]

# Create the plot
plt.figure(figsize=(8, 6))
plt.plot(x, radar, marker='o', linestyle='-', color='b', label='Radar Model')
plt.plot(x, satellite, marker='o', linestyle='--', color='g', label='Radar & Satellite Model')

# Add titles and labels
plt.title("Violent rain")
plt.xlabel("Lead Time (min)")
plt.ylabel("CSI Value")
plt.legend()  # Show the legend
plt.grid(True)
# Adjust layout and save the plot
plt.tight_layout()
plt.savefig('Violent rain lead time.png')