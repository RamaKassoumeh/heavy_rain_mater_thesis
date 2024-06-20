import torch
import torch.nn as nn


class RainNet(nn.Module):
    
    def __init__(self, 
                 mode = "regression",
                conv_shape = [["1", [6,64]],
                        ["2" , [64,128]],
                        ["3" , [128,256]],
                        ["4" , [256,512]],
                        ["5" , [1536,512]],
                        ["6" , [768,256]],
                        ["7" , [256,384]],
                        ["8" , [384,128]],
                        ["9" , [128,64]]],
                kernel_size_radar =  (1, 3, 3),
                kernel_size_satellite = {
                        '1': (11, 3, 3),
                        '2': (5, 3, 3),
                        '3': (2, 3, 3),
                        '4': (1, 3, 3)
                        }):
        
        super().__init__()
        self.kernel_size_radar = kernel_size_radar
        self.kernel_size_satellite = kernel_size_satellite
        self.mode = mode

        self.conv = nn.ModuleDict()
        for name, (in_ch, out_ch) in conv_shape:
            if name != "7":
                self.conv[name] = self.make_conv_block(in_ch,
                                                       out_ch ,self.kernel_size_radar)
            else:
                self.conv[name] = nn.Sequential(
                    self.make_conv_block(in_ch, out_ch, self.kernel_size_radar),
                    nn.Conv3d(out_ch, 2, (1,3,3), padding='same'))
        
        
        self.pool_radar = nn.MaxPool3d(kernel_size = (1,2,2))
        self.pool_satellite = nn.MaxPool3d(kernel_size = (2,2,2))
        self.upsample_1st = nn.Upsample(scale_factor=(3,2,2))
        self.upsample = nn.Upsample(scale_factor=(2,2,2))
        self.drop = nn.Dropout(p=0.5)
        #Ayzel et al. uses basic dropout
        ##self.drop = nn.Dropout2d(p=0.5)
        
        if self.mode == "regression":
            self.last_layer = nn.Sequential(
                nn.Conv3d(2, 1, kernel_size=(12,1,1), padding = 'valid'),
                )
        elif self.mode == "segmentation":
            self.last_layer = nn.Sequential(
                nn.Conv3d(2, 1, kernel_size=(12,1,1), padding = 'valid'),
                nn.Sigmoid())
        else:
            raise NotImplementedError()
            
    def make_conv_block(self, in_ch, out_ch, kernel_size_radar):
        
        return nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size_radar, padding='same'),
            nn.ReLU(),
            nn.Conv3d(out_ch, out_ch, kernel_size_radar, padding='same'),
            nn.ReLU()
            )
    
        
    def forward(self, x):
        x_radar=x[:,:,0:1,:,:]
        x_satellite=x[:,:,1:,:,:]
        x1r = self.conv["1"](x_radar.float()) # conv1s
        x2r = self.conv["2"](self.pool_radar(x1r)) # conv2s
        x3r = self.conv["3"](self.pool_radar(x2r)) # conv3s
        x_r = self.conv["4"](self.pool_radar(x3r)) # conv4s

        x1s = self.conv["1"](x_satellite.float()) # conv1s
        x2s = self.conv["2"](self.pool_satellite(x1s)) # conv2s
        x3s = self.conv["3"](self.pool_satellite(x2s)) # conv3s
        x_s = self.conv["4"](self.pool_satellite(x3s)) # conv4s
        # x = self.conv["5"](self.pool(self.drop(x4s))) # conv5s
        x = torch.cat((self.upsample_1st(self.drop(x)), x3s), dim=1) # up6
        x = torch.cat((self.upsample(self.conv["5"](x)), x2s), dim=1) # up7
        x = torch.cat((self.upsample(self.conv["6"](x)), x1s), dim=1) # up8
        # x = torch.cat((self.upsample(self.conv["8"](x)), x1s), dim=1) # up9
        x = self.conv["7"](x) #conv9
        x = self.last_layer(x) #outputs
        
        return x