from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from plotting import plot_images

from torch.utils.data import DataLoader, Subset
import os
from PIL import Image

import numpy as np
from sklearn.metrics import confusion_matrix

from tqdm import tqdm

from test_metrics import calculate_metrics,categories_threshold

def train_model(train_dataset,validate_data,test_data,inverseTransform,model,file_name):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=10,
        shuffle=True
    )

    validate_loader = DataLoader(
        dataset=validate_data,
        batch_size=5,
        shuffle=True
    )
    test_loader = DataLoader(
        dataset=test_data,
        batch_size=5,
        shuffle=True
    )

    class LogCoshLoss(nn.Module):
        def __init__(self):
            super(LogCoshLoss, self).__init__()

        def forward(self, y_pred, y_true):
            # Compute the difference between predicted and true values
            diff = y_pred - y_true
            # Compute log(cosh(x)) element-wise
            loss = torch.log(torch.cosh(diff))
            # Compute the mean loss over all examples
            loss = torch.mean(loss)
            return loss
        
    class LogCoshThresholdLoss(nn.Module):
        def __init__(self,lower_threshold,upper_threshold):
            super(LogCoshThresholdLoss, self).__init__()
            self.lower_threshold=lower_threshold.cuda()
            self.upper_threshold=upper_threshold.cuda()

        def forward(self, y_pred, y_true):
            # get data between threasholds

            # Create a boolean mask for values between the thresholds
            mask = (y_true >= self.lower_threshold) & (y_true < self.upper_threshold)

            # Use the boolean mask to filter the tensor
            y_true_filtered_values = y_true[mask[0,0]]
            y_pred_filtered_values = y_pred[mask[0,0]]
            # Compute the difference between predicted and true values
            diff = y_pred_filtered_values - y_true_filtered_values
            # Compute log(cosh(x)) element-wise
            loss = torch.log(torch.cosh(diff))
            # Compute the mean loss over all examples``
            loss = torch.mean(loss)
            return loss

    # Get a batch
    # input, _ = next(iter(validate_loader))

    # model=RainNet()
    no_param=sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"number of parameters in the model is {no_param}")
    model=torch.nn.DataParallel(model)
    model.cuda()
    # optim = Adam(model.parameters(), lr=1e-4)
    optim = Adam(model.parameters(), lr=0.1)
    # Define learning rate scheduler
    # scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=5, gamma=0.1)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[10,6,4], gamma=0.1)
    # Binary Cross Entropy, target pixel values either 0 or 1
    # criterion = nn.BCELoss(reduction='sum')
    # -999 -0
    # criterion_undefined_rain = LogCoshThresholdLoss(transform(np.array([[-999]])),transform(np.array([[0]])))
    # # 0-2.5
    # criterion_light_rain = LogCoshThresholdLoss(transform(np.array([[0]])),transform(np.array([[2.5]])))
    # # 2.5 - 7.5
    # criterion_moderate_rain = LogCoshThresholdLoss(transform(np.array([[2.5]])),transform(np.array([[7.5]])))
    # # 73.5-200
    # criterion_heavy_rain = LogCoshThresholdLoss(transform(np.array([[7.5]])),transform(np.array([[201]])))
    num_epochs = 50
    criterion = LogCoshLoss()
    # Initializing in a separate cell, so we can easily add more epochs to the same run
    # Calculate CSI for each category across all images
    csi_values = {category: [] for category in categories_threshold.keys()}

    # Calculate fss for each category across all images
    fss_values = {category: [] for category in categories_threshold.keys()}
    writer = SummaryWriter(f'runs/{file_name}_{timestamp}')
    start_epoch=1
    checkpoint_path =f'models/{file_name}_model_checlpoint.pth'
    # Load the checkpoint if it exists
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from checkpoint at epoch {start_epoch}")

    for epoch in range(start_epoch, num_epochs + 1):

        train_loss = 0
        acc=0
        total =0
        model.train() 
        for batch_num, (input, target) in enumerate(tqdm(train_dataloader), 1):
            optim.zero_grad()
            output = model(input)
            output_flatten=output.flatten()
            target_flatten=target.flatten()
            
            # loss_undefined_rain = criterion_undefined_rain(output_flatten, target_flatten)
            # loss_light_rain = criterion_light_rain(output_flatten, target_flatten)
            # loss_moderate_rain = criterion_moderate_rain(output_flatten, target_flatten)
            # loss_heavy_rain = criterion_heavy_rain(output_flatten, target_flatten)
            # loss=0.1*loss_undefined_rain+0.2*loss_light_rain+0.3*loss_moderate_rain+0.4*loss_heavy_rain
            loss = criterion(output_flatten, target_flatten)
            loss.backward()
            optim.step()
            train_loss += loss.item()
            if batch_num%100 ==0:
                target=inverseTransform(target)
                input=inverseTransform(input)
                output=inverseTransform(output)
                # plot_images([input[0,0,input.shape[2]-1],input[0,0,input.shape[2]-2],input[0,0,input.shape[2]-3],input[0,0,input.shape[2]-4],input[0,0,input.shape[2]-5],input[0,0,input.shape[2]-6] ,target[0][0],output[0][0]], 2, 4,epoch,batch_num,'train',file_name)
                plot_images([input[0,input.shape[1]-1],input[0,input.shape[1]-2],input[0,input.shape[1]-3],input[0,input.shape[1]-4],input[0,input.shape[1]-5],input[0,input.shape[1]-6] ,target[0,0],output[0,0]], 2, 4,epoch,batch_num,'train',file_name)

        train_loss /= len(train_dataloader.dataset)
        print(f"the train loss is {train_loss}")
        val_loss = 0
        model.eval()
        csi_values.clear()
        fss_values.clear()
        with torch.no_grad():
            for batch_num, (input, target) in enumerate(tqdm(validate_loader), 1):
                output = model(input)
                output_flatten=output.flatten()
                target_flatten=target.flatten()
                # loss_undefined_rain = criterion_undefined_rain(output_flatten, target_flatten)
                # loss_light_rain = criterion_light_rain(output_flatten, target_flatten)
                # loss_moderate_rain = criterion_moderate_rain(output_flatten, target_flatten)
                # loss_heavy_rain = criterion_heavy_rain(output_flatten, target_flatten)
                # loss=0.1*loss_undefined_rain+0.2*loss_light_rain+0.3*loss_moderate_rain+0.4*loss_heavy_rain
                loss = criterion(output_flatten, target_flatten)
                val_loss += loss.item()
                if batch_num%100 ==0:
                    target=inverseTransform(target)
                    input=inverseTransform(input)
                    output=inverseTransform(output)
                    # plot_images([input[0,0,input.shape[2]-1],input[0,0,input.shape[2]-2],input[0,0,input.shape[2]-3] ,input[0,0,input.shape[2]-4] ,input[0,0,input.shape[2]-5] ,input[0,0,input.shape[2]-6]  ,target[0][0] ,output[0][0] ], 2, 4,epoch,batch_num,'validate',file_name)
                    plot_images([input[0,input.shape[1]-1],input[0,input.shape[1]-2],input[0,input.shape[1]-3],input[0,input.shape[1]-4],input[0,input.shape[1]-5],input[0,input.shape[1]-6] ,target[0,0],output[0,0]], 2, 4,epoch,batch_num,'validate',file_name)
                if epoch%5==0 and batch_num%10 ==0:
                    mse,csi,fss=calculate_metrics(target,output)
                    for category in categories_threshold.keys():
                        csi_values[category].append(csi[category])
                        fss_values[category].append(fss[category])
        if epoch%5==0:
            mse,csi,fss=calculate_metrics(target,output)
            for category in categories_threshold.keys():
                    csi_values[category].append(csi[category])
                    fss_values[category].append(fss[category])
            csi_means = {category: np.nanmean(csi_values[category]) for category in categories_threshold.keys()}
            fss_means = {category: np.nanmean(fss_values[category]) for category in categories_threshold.keys()}
            writer.add_scalars(f'CSI values',csi_means,epoch)
            writer.add_scalars(f'FSS values',fss_means,epoch)



        val_loss /= len(validate_loader.dataset)
        # val_loss /= 128
        print(f"the validate loss is {val_loss}")
        print("Epoch:{} Training Loss:{:.2f} Validation Loss:{:.2f}\n".format(
            epoch, train_loss, val_loss))

        # Log the running loss averaged per batch for both training and validation
        writer.add_scalars('Training vs. Validation Loss',
                        {'Training': train_loss, 'Validation': val_loss},
                        epoch)
        writer.flush()
        # Save checkpoint every 5 epochs
        if epoch% 5 == 0:
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optim.state_dict(),
                }, checkpoint_path)
            print(f'Checkpoint saved at epoch {epoch + 1}')

    # Save the model's state dictionary
    torch.save(model.state_dict(), f'models/{file_name}_model.pth')

    # Save the optimizer's state dictionary if needed
    torch.save(optim.state_dict(), f'models/{file_name}_optimizer.pth')

    # Calculate RMSE for each image
    rmse_values = []

    # Calculate CSI for each category across all images
    csi_values = {category: [] for category in categories_threshold.keys()}
    output_file_path = file_name+'_results.txt'  # Specify the file path where you want to save the results

    model.eval()
    csi_values.clear()
    fss_values.clear()
    with torch.no_grad():
        for batch_num, (input, target) in enumerate(tqdm(test_loader), 1):
            output = model(input)
            actual_img=inverseTransform(target)
            predicted_img=inverseTransform(output)
            if batch_num%100 ==0:
                input=inverseTransform(input)
                # plot_images([input[0,0,input.shape[2]-1],input[0,0,input.shape[2]-2],input[0,0,input.shape[2]-3],input[0,0,input.shape[2]-4],input[0,0,input.shape[2]-5],input[0,0,input.shape[2]-6] ,target[0][0],output[0][0]], 2, 4,epoch,batch_num,'train',file_name)
                plot_images([input[0,input.shape[1]-1],input[0,input.shape[1]-2],input[0,input.shape[1]-3],input[0,input.shape[1]-4],input[0,input.shape[1]-5],input[0,input.shape[1]-6] ,actual_img[0,0],predicted_img[0,0]], 2, 4,1,batch_num,'test',file_name)
            
            mse,csi,fss=calculate_metrics(target,output)
            for category in categories_threshold.keys():
                csi_values[category].append(csi[category])
                fss_values[category].append(fss[category])
            # Append RMSE to list
            rmse_values.append(mse)


    # Calculate the average RMSE across all images
    average_rmse = np.mean(rmse_values)

    # Display the results
    print(f"Average RMSE across all images: {average_rmse}")
    with open(f'results/{output_file_path}', 'w') as file:
        file.write(f"\nAverage RMSE across all images: {average_rmse}\n")

        # Calculate the average CSI for each category across all images
        average_csi = {category: np.mean(csi_values[category]) for category in categories_threshold.keys()}

        # Display the results
        print("Average CSI for each category across all images:")
        for category, avg_csi in average_csi.items():
            print(f"{category}: {avg_csi}")
            file.write(f"\nAverage CSI for category: {category}: {avg_csi}\n")
        file.close()

