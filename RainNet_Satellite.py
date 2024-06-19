import torch
import torch.nn as nn


class RainNet(nn.Module):
    
    def __init__(self, 
                 kernel_size = 3,
                 mode = "regression",
                #  im_shape = (256,256),
                 conv_shape = [["1", [6,64]],
                        ["2" , [64,128]],
                        ["3" , [128,256]],
                        ["4" , [256,512]],
                        ["5" , [512,1024]],
                        ["6" , [1536,512]],
                        ["7" , [768,256]],
                        ["8" , [384,128]],
                        ["9" , [192,64]]],
                encoder_radar = [["1", [6,64]],
                        ["2" , [64,128]],
                        ["3" , [128,256]],
                        ["4" , [256,512]],
                        ["5" , [512,1024]]],
                encoder_satellite = [["1", [6*11,64*11]],
                        ["2" , [64*11,128*11]],
                        ["3" , [128*11,256*11]],
                        ["4" , [256*11,512*11]],
                        ["5" , [512*11,1024*11]]],
                decoder = [["6" , [1024+512+1024*11+512*11,512+512*11]],
                        ["7" , [512+256+512*11+256*11,256+256*11]],
                        ["8" , [256+128+256*11+128*11,128+128*11]],
                        ["9" , [128+64+128*11+64*11,64+64*11]]]):
        
        super().__init__()
        self.kernel_size = kernel_size
        self.mode = mode

        self.conv_encoder_radar = nn.ModuleDict()
        for name, (in_ch, out_ch) in encoder_radar:
            self.conv_encoder_radar[name] = self.make_conv_block(in_ch,
                                                       out_ch ,self.kernel_size)

        self.conv_encoder_sat = nn.ModuleDict()
        for name, (in_ch, out_ch) in encoder_satellite:
            self.conv_encoder_sat[name] = self.make_conv_block(in_ch,
                                                       out_ch ,self.kernel_size)
        self.conv_decoder = nn.ModuleDict()    
        for name, (in_ch, out_ch) in decoder:
            if name != "9":
                self.conv_decoder[name] = self.make_conv_block(in_ch,
                                                       out_ch ,self.kernel_size)
            else:
                self.conv_decoder[name] = nn.Sequential(
                    self.make_conv_block(in_ch, out_ch, self.kernel_size),
                    nn.Conv2d(out_ch, 2, 3, padding='same'))        
        
        self.pool = nn.MaxPool2d(kernel_size = (2,2))
        self.upsample = nn.Upsample(scale_factor=(2,2))
        self.drop = nn.Dropout(p=0.5)
        #Ayzel et al. uses basic dropout
        ##self.drop = nn.Dropout2d(p=0.5)
        
        if self.mode == "regression":
            self.last_layer = nn.Sequential(
                nn.Conv2d(2, 1, kernel_size=1, padding = 'valid'),
                )
        elif self.mode == "segmentation":
            self.last_layer = nn.Sequential(
                nn.Conv2d(2, 1, kernel_size=1, padding = 'valid'),
                nn.Sigmoid())
        else:
            raise NotImplementedError()
            
    def make_conv_block(self, in_ch, out_ch, kernel_size):
        
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding='same'),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size, padding='same'),
            nn.ReLU()
            )
    
        
    def forward(self, x_radar,x_sat):
        x1r = self.conv_encoder_radar["1"](x_radar.float()) # conv1s
        x2r = self.conv_encoder_radar["2"](self.pool(x1r)) # conv2s
        x3r = self.conv_encoder_radar["3"](self.pool(x2r)) # conv3s
        x4r = self.conv_encoder_radar["4"](self.pool(x3r)) # conv4s
        x_r = self.conv_encoder_radar["5"](self.pool(self.drop(x4r))) # conv5s

        x1s = self.conv_encoder_sat["1"](x_sat.float()) # conv1s
        x2s = self.conv_encoder_sat["2"](self.pool(x1s)) # conv2s
        x3s = self.conv_encoder_sat["3"](self.pool(x2s)) # conv3s
        x4s = self.conv_encoder_sat["4"](self.pool(x3s)) # conv4s
        x_s = self.conv_encoder_sat["5"](self.pool(self.drop(x4s))) # conv5s

        x=torch.cat((x_r,x_s),dim=1) # concatenate radar and satellite data

        x = torch.cat((self.upsample(self.drop(x)), x4r,x4s), dim=1) # up6
        x = torch.cat((self.upsample(self.conv_decoder["6"](x)),x3r, x3s), dim=1) # up7
        x = torch.cat((self.upsample(self.conv_decoder["7"](x)),x2r, x2s), dim=1) # up8
        x = torch.cat((self.upsample(self.conv_decoder["8"](x)),x1r ,x1s), dim=1) # up9
        x = self.conv_decoder["9"](x) #conv9
        x = self.last_layer(x) #outputs
        
        return x