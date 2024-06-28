
from datetime import datetime

import torch
from RadarFilterImageDataset import RadarFilterImageDataset
from RadarFilterRainNetDataset import RadarFilterRainNetDataset
from RadarFilterRainNetSatelliteDataset import RadarFilterRainNetSatelliteDataset
from plotting import plot_images,plot_image
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
from torchvision import transforms

min_value=0
max_value=10

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

def plot_Img(image_list, epoch,batch_num,name):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for i in range(1):
        for j in range(2):
            image=image_list[i * 2 + j]
            image = image.detach().cpu().numpy()
            axes[j].imshow(image[0,0])
            axes[j].axis('off')
    plt.subplots_adjust(wspace=0.1, hspace=0.1)  # Adjust spacing between subplots
    isExist = os.path.exists(f"output/test_visualization_{timestamp}")
    if not isExist:
        os.mkdir(f"output/test_visualization_{timestamp}") 

    plt.savefig(f"output/test_visualization_{timestamp}/{name}_{epoch}_{batch_num}")
    plt.close()

mean=0.21695
std=0.9829
# make transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Lambda(lambda x: x.unsqueeze(0))  ,# Add a new dimension at position 0
    # transforms.Lambda(lambda x: x.cuda()) , # send data to cuda
    # transforms.Normalize(mean=[mean,],
    #                          std=[std,],)
    # transforms.Lambda(lambda x: (x-min_value)/(max_value-min_value)),
    # transforms.Lambda(lambda x: torch.log2(x+1))
    transforms.Lambda(lambda x:  torch.log(x+1) / torch.log(torch.tensor(10.0)))
    
])
inverseTransform= transforms.Compose([
    # transforms.Lambda(lambda x: x.unsqueeze(0))  ,# Add a new dimension at position 0
    # transforms.Lambda(lambda x: x.cuda()) , # send data to cuda
    # transforms.Normalize(mean=[-mean/std,],
                            #  std=[1/std,])
    transforms.Lambda(lambda x: torch.pow(10, x)-1),
    # transforms.Lambda(lambda x: (x*(max_value - min_value))+min_value)
    
])
transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Lambda(lambda x: x.unsqueeze(0))  ,# Add a new dimension at position 0
    # transforms.Lambda(lambda x: x.cuda()) , # send data to cuda
    # transforms.Normalize(mean=[mean,],
    #                          std=[std,],)
    # transforms.Lambda(lambda x: (x-min_value)/(max_value-min_value)),
    # transforms.Lambda(lambda x: torch.log2(x+1))
    transforms.Lambda(lambda x: torch.log(x+0.01)),
    transforms.Lambda(lambda x: x.float())
    
])
inverseTransform= transforms.Compose([
    # transforms.Lambda(lambda x: x.unsqueeze(0))  ,# Add a new dimension at position 0
    # transforms.Lambda(lambda x: x.cuda()) , # send data to cuda
    # transforms.Normalize(mean=[-mean/std,],
                            #  std=[1/std,])
    transforms.Lambda(lambda x: torch.exp(x)-0.01),
    # transforms.Lambda(lambda x: (x*(max_value - min_value))+min_value)
    transforms.Lambda(lambda x: x) 
])

def custom_transform1(x):
    # Use PyTorch's where function to apply the transformation element-wise
    return torch.where(x >= 0, x + 1, x)
def custom_transform2(x):
    # Use PyTorch's where function to apply the transformation element-wise
    return torch.where(x < 0, 0, x)


transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Lambda(lambda x: x.unsqueeze(0))  ,# Add a new dimension at position 0
    # transforms.Lambda(lambda x: x.cuda()) , # send data to cuda
    # transforms.Normalize(mean=[mean,],
    #                          std=[std,],)
    # transforms.Lambda(lambda x: (x-min_value)/(max_value-min_value)),
    # transforms.Lambda(lambda x: torch.log2(x+1))
    transforms.Lambda(custom_transform1) ,
    transforms.Lambda(custom_transform2) ,
    # transforms.Lambda(lambda x: torch.log(x+1)),
     transforms.Lambda(lambda x:  (torch.log(x+1) / torch.log(torch.tensor(max_value))).float()),
    # transforms.Lambda(lambda x: x.float())
    
])

def invert_custom_transform1(x):
    # Use PyTorch's where function to apply the transformation element-wise
    return torch.where(x > -0, x-1, x)
def invert_custom_transform2(x):
    # Use PyTorch's where function to apply the transformation element-wise
    return torch.where(x <= -0, -999, x) 

inverseTransform= transforms.Compose([
    # transforms.Lambda(lambda x: x.unsqueeze(0))  ,# Add a new dimension at position 0
    # transforms.Lambda(lambda x: x.cuda()) , # send data to cuda
    # transforms.Normalize(mean=[-mean/std,],
                            #  std=[1/std,])
    # transforms.Lambda(lambda x: torch.exp(x)-1),
    transforms.Lambda(lambda x: torch.pow(max_value, x)-1),
    transforms.Lambda(invert_custom_transform2) ,
    transforms.Lambda(invert_custom_transform1) ,
    
    # transforms.Lambda(lambda x: (x*(max_value - min_value))+min_value)
    # transforms.Lambda(lambda x: x) 
])
train_dataset = RadarFilterRainNetSatelliteDataset(
    img_dir='../RadarTest',
    sat_dir='../SatelliteData',
    return_original=True,
    transform=transform,
    inverse_transform=inverseTransform
)


train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=1,
    shuffle=True
)


counter=0
for batch_num, (input, target,original_target) in enumerate(train_dataloader, 1):
    counter+=1
    target=inverseTransform(target)
    input=inverseTransform(input)
    target_np=target.detach().cpu().numpy()
    # plot_images([input[0,0,input.shape[2]-1],input[0,0,input.shape[2]-2],input[0,0,input.shape[2]-3],input[0,0,input.shape[2]-4],input[0,0,input.shape[2]-5],input[0,0,input.shape[2]-6] ,target[0][0],original_target[0][0]],2, 4,1,batch_num,'test',"test_visualization")
    # plot_images([input[0,input.shape[1]-1,0],input[0,input.shape[1]-2,0],input[0,input.shape[1]-1,0],input[0,input.shape[1]-2,0],input[0,input.shape[1]-1,0],input[0,input.shape[1]-2,0] ,target[0][0],original_target[0][0]],2, 4,1,batch_num,'test',"test_visualization")
    plot_images([input[0,input.shape[1]-1],input[0,input.shape[1]-2],input[0,input.shape[1]-1],input[0,input.shape[1]-2],input[0,input.shape[1]-3],input[0,input.shape[1]-4] ,target[0][0],original_target[0][0]],2, 4,1,batch_num,'test',"test_visualization")
    plot_image(input[0,input.shape[1]-1])
    if counter >=100:
        break

# # plot normalize image 
# counter=0
# for batch_num, (input, target,original_target) in enumerate(train_dataloader, 1):
#     counter+=1
#     target=inverseTransform(target)
#     target_np=target.detach().cpu().numpy()
#     if (target_np > 10).sum()>0:
#         ind=np.argwhere(target_np>200)
#     plot_images([input[0,0,input.shape[2]-1],input[0,0,input.shape[2]-2],input[0,0,input.shape[2]-3],input[0,0,input.shape[2]-4],input[0,0,input.shape[2]-5],input[0,0,input.shape[2]-6] ,target[0][0],original_target[0][0]],2, 4,1,batch_num,'test',"test_visualization")

#     plot_Img([original_target,target],1,batch_num,'test_plot')
#     if counter >=100:
#         break