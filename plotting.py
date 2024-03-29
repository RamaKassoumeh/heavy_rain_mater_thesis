import matplotlib.pyplot as plt
from matplotlib import patches
from datetime import datetime
import numpy as np
import os

# Define the ranges and corresponding colors
ranges = [
    (-999, -0.1, 'white'),
    (0, 0.06, 'lightgrey'),
    (0.06, 0.12, 'cyan'),
    (0.12, 0.21, 'dodgerblue'),
    (0.21, 0.36, 'blue'),
    (0.36, 0.65, 'LimeGreen'),
    (0.65, 1.15, 'forestgreen'),
    (1.15, 2.05, 'green'),
    (2.05, 3.65, 'yellow'),
    (3.65, 6.48, 'gold'),
    (6.48, 11.53, 'darkorange'),
    (11.53, 20.5, 'red'),
    (20.5, 36.46, 'crimson'),
    (36.46, 64.84, 'darkred'),
    (64.84, 115.31, 'violet'),
    (115.31, 205.05, 'magenta'),
    (205.05, 1000, 'black')
]
bounds = [start for start, _, _ in ranges]
midpoints = [(start + end) / 2 for start, end, _ in ranges]

# Map each value to a color based on the defined ranges
def map_value_to_color(value):
    for start, end, color in ranges:
        if start <= round(value,2) <= end:
            return color
    return 'black'  # Return black if value doesn't fall in any range

num_points = int(2000/ 0.01)

# Create a color map from the function
cmap = plt.cm.colors.ListedColormap([map_value_to_color(value) for value in np.linspace(-999, 1000, num_points)])
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')


def plot_images(image_list, row, col, epoch, batch_num, name):
    # Create a figure and divide it into two areas
    fig = plt.figure(figsize=(12, 6))  # Set overall figure size
    gs = fig.add_gridspec(1, 2, width_ratios=[5, 1])  # Divide into two areas, one with 3 times width

    # Create subplots for the smaller area to plot rectangles
    ax_smaller = fig.add_subplot(gs[1])

    # Plot rectangles with colors in the smaller area
    for i, (start, end, color) in enumerate(ranges):
        rect = patches.Rectangle((0, i), 6, 1, linewidth=1, facecolor=color)
        ax_smaller.add_patch(rect)
        ax_smaller.text(1, i + 0.5, f'{start}-{end}' if start>=0 else 'undefined', verticalalignment='center', horizontalalignment='center',
                        color='white' if i > 9 else 'black')

    ax_smaller.set_xlim(0, 2)  # Adjust xlim to fit rectangles and text
    ax_smaller.set_ylim(0, len(ranges))  # Adjust ylim to fit rectangles and text
    ax_smaller.axis('off')  # Turn off axis

    # Create subplots for the larger area to display 8 images
    axs_larger = fig.add_subplot(gs[0])
    axs_larger.set_xticks([])  # Remove x ticks
    axs_larger.set_yticks([])  # Remove y ticks
    axs_larger.axis('off')  # Turn off axis
    # Plot images in the larger area

    # Create a grid of subplots within the larger subplot
    inner_grid = np.zeros((row, col), dtype=object)
    for i in range(row):
        for j in range(col):
            inner_grid[i, j] = fig.add_subplot(gs[0].subgridspec(row, col)[i, j])

    # for ax, image, title in zip(inner_grid.flat, image_list, [f'Image {i}' for i in range(1, len(image_list)+1)]):
    #     # image = image_list[i * col + j]
    #     image = image.detach().cpu().numpy()
    #     # image = image.astype(np.uint8)
    #     ax.imshow(image, cmap=cmap, vmin=0, vmax=200)
    #     ax.axis('off')

    ax_grid = inner_grid.flatten()

    for i in range(len(image_list)):
        image = image_list[i]
        image = image.detach().cpu().numpy()
        ax_grid[i].imshow(image, cmap=cmap, vmin=-999, vmax=1000)  # Assuming grayscale images
        ax_grid[i].axis('off')
        if i < 6: 
            ax_grid[i].set_title(f't - {(5 - i) * 5} mins' if i != 5 else 't mins')
        elif i == 6:
            ax_grid[i].set_title(f't +5 mins (target)')

        elif i == 7:
            ax_grid[i].set_title(f't +5 mins (predict)')
    plt.tight_layout()
    # plt.show()
    isExist = os.path.exists(f"output/image_radar_trainer_128_128_30M_filter_data_{timestamp}")
    if not isExist:
        os.mkdir(f"output/image_radar_trainer_128_128_30M_filter_data_{timestamp}") 

    plt.savefig(f"output/image_radar_trainer_128_128_30M_filter_data_{timestamp}/{name}_{epoch}_{batch_num}")
