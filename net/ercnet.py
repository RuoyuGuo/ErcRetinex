import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
from net.transformer import EnhanceNet as v4

class Conv5net(nn.Module):
    def __init__(self, outc, layer=5, num=64):
        super(Conv5net, self).__init__()
        net = []
        net.append(nn.ReflectionPad2d(1))
        net.append(nn.Conv2d(3, num, 3, 1, 0))
        net.append(nn.ReLU())

        for _ in range(layer):
            net.append(nn.ReflectionPad2d(1))
            net.append(nn.Conv2d(num, num, 3, 1, 0))
            net.append(nn.ReLU())

        net.append(nn.ReflectionPad2d(1))
        net.append(nn.Conv2d(num, outc, 3, 1, 0))

        self.net = nn.Sequential(*net)

    def forward(self, input):
        return torch.sigmoid(self.net(input))


class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()        
        self.R_net = Conv5net(outc=3, layer=3)
        self.L_net = Conv5net(outc=1, layer=3)
        
        self.enhancenet = v4(in_ch=3, base_ch=64, guide_ch=4, num_enblocks=[1,2,4], num_deblocks=[2, 2])
        
    def forward(self, input):
        x = input

        L = self.L_net(x)
        R = self.R_net(x)
        
        R_clean, _ = self.enhancenet(R, torch.cat([x, L], dim=1))

        return L, R, R_clean
    
    def get_feature(self, input):
        x = input

        L = self.L_net(x)
        R = self.R_net(x)
        
        _, feature = self.enhancenet(R, torch.cat([x, L], dim=1))

        return feature
    
    def get_lg_feature(self, input):
        
        x = input

        L = self.L_net(x)
        R = self.R_net(x)
        
        encoder_feature, multi_guide_fea, x_g, x_l = self.enhancenet.get_lg_feature(R, torch.cat([x, L], dim=1))

        return encoder_feature, multi_guide_fea, x_g, x_l