import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
parparent = os.path.dirname(parent)
sys.path.append(current)
sys.path.append(parent)
sys.path.append(parparent)

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np

# Paths to the directories for the two runs
run_file_heavy_rain = f'{parparent}/runs/radar_trainer_30M_RainNet_3d_Log_summer_15_min_20240819_211329/CSI values_heavy rain'
run_file_violent_rain = f'{parparent}/runs/radar_trainer_30M_RainNet_3d_Log_summer_15_min_20240819_211329/CSI values_Violent rain'

# Load the event files
event_acc_heavy = EventAccumulator(run_file_heavy_rain)
event_acc_violent = EventAccumulator(run_file_violent_rain)
event_acc_heavy.Reload()
event_acc_violent.Reload()

# Assuming both runs logged the same tags and epochs
scalar_tags_heavy = event_acc_heavy.Tags()['scalars']
scalar_tags_violent = event_acc_violent.Tags()['scalars']

# We assume the tags are the same in both directories and extract scalar values
scalars_heavy = event_acc_heavy.Scalars(scalar_tags_heavy[0])  # Adjust if multiple tags
scalars_violent = event_acc_violent.Scalars(scalar_tags_violent[0])

# Extract epochs and values
epochs = [scalar.step for scalar in scalars_heavy]
values_heavy = np.array([scalar.value for scalar in scalars_heavy])
values_violent = np.array([scalar.value for scalar in scalars_violent])
# values_heavy[19]=0

# Calculate composite score (e.g., sum or average of the two)
composite_scores = (values_heavy + values_violent)  # or (values_heavy + values_violent) / 2 for average

# Identify the best epoch based on the composite score
best_epoch_index = np.argmax(composite_scores)
best_epoch = epochs[best_epoch_index]

print(f"The best epoch based on the composite score is: {best_epoch}")
print(f"Composite score at best epoch: {composite_scores[best_epoch_index]}")

best_epoch_index = np.argmax(values_heavy)
best_epoch = epochs[best_epoch_index]
print(f"The best epoch based on the composite score is: {best_epoch}")
print(f"Composite score at best epoch: {composite_scores[best_epoch_index]}")