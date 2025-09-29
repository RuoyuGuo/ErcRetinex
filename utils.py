import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import random

def std(img, window_size=7):
    assert window_size % 2 == 1
    pad = window_size // 2

    # calculate std on the mean image of the color channels
    img = torch.mean(img, dim=1, keepdim=True)
    N, C, H, W = img.shape
    img = nn.functional.pad(img, [pad] * 4, mode='reflect')
    img = nn.functional.unfold(img, kernel_size=window_size)
    img = img.view(N, C, window_size * window_size, H, W)
    img = img - torch.mean(img, dim=2, keepdim=True)
    img = img * img
    img = torch.mean(img, dim=2, keepdim=True)
    img = torch.sqrt(img)
    img = img.squeeze(2)
    
    # img = norm(img)
    
    return img

def GradMean(img, window_size=7):
    assert window_size % 2 == 1
    pad = window_size // 2

    # calculate std on the mean image of the color channels
    img = torch.mean(img, dim=1, keepdim=True)
    N, C, H, W = img.shape
    img = nn.functional.pad(img, [pad] * 4, mode='reflect')
    img = nn.functional.unfold(img, kernel_size=window_size)
    img = img.view(N, C, window_size * window_size, H, W)
    img = torch.mean(img, dim=2, keepdim=True)
    img = img.squeeze(2)
    
    # img = norm(img)
    
    return img

def norm(img):
    B, C, H, W = img.shape
    
    min_v,_ = torch.min(img.view(B,-1), dim=1, keepdim=True)
    max_v,_ = torch.max(img.view(B,-1), dim=1, keepdim=True)
    
    min_v = min_v.unsqueeze(2).unsqueeze(2)
    max_v = max_v.unsqueeze(2).unsqueeze(2)
    
    img = (img - min_v) / (max_v - min_v)
    
    return img
    
def generate_Md(input, beta1, beta2, mid, lower, upper, window_size=7):
    N, C, H, W = input.shape
    ratio = input.new_ones((N, 1, H, W)) * mid
    input_std = norm(std(input, window_size=window_size))
    ratio[input_std < lower] = torch.sigmoid((input_std - lower)*beta1)[input_std < lower]
    ratio[input_std > upper] = torch.sigmoid((input_std - upper)*beta2)[input_std > upper]
    ratio = ratio.detach()

    return ratio, input_std


def piecewise_norm(input_std, beta1, beta2, mid, lower, upper):
    N, C, H, W = input_std.shape
    ratio = input_std.new_ones((N, 1, H, W)) * mid
    ratio[input_std < lower] = torch.sigmoid((input_std - lower)*beta1)[input_std < lower]
    ratio[input_std > upper] = torch.sigmoid((input_std - upper)*beta2)[input_std > upper]
    ratio = ratio.detach()

    return ratio


class sobelFilter(nn.Module):
    '''
    apply sobel filter on input tensor
    output tensor with 2 channels
    '''

    def __init__(self):
        super(sobelFilter, self).__init__()

        self.sobel_x = torch.tensor([[1., 0., -1.], \
                            [2., 0., -2.], \
                            [1., 0., -1.]], dtype=torch.float32).unsqueeze(0).expand(3,1,3,3).cuda()

        self.sobel_y = torch.tensor([[1.,   2.,  1.],\
                            [0.,   0.,  0.],\
                            [-1., -2., -1.]], dtype=torch.float32).unsqueeze(0).expand(3,1,3,3).cuda()
        
    def forward(self, x):
        grad_out = F.pad(x, (1, 1, 1, 1), 'reflect')
        grad_out_x = F.conv2d(grad_out, self.sobel_x, groups=3)
        grad_out_y = F.conv2d(grad_out, self.sobel_y, groups=3)

        return grad_out_x, grad_out_y
    

class LossStage1(nn.Module):
    def __init__(self):
        super(LossStage1, self).__init__()
        self.L2 = torch.nn.MSELoss()
        self.L1 = torch.nn.L1Loss()
        
    def forward(self, L1, L2, 
                R1, R2, 
                R1_clean, R2_clean, 
                im1, im2):
        #r consistency
        c_loss = 0
        c_loss += self.L2(R1_clean, R2_clean)
        c_loss += self.L2(R1_clean, R1)
        
        #r and l regularisation    (L1.detach() + 1e-8)
        max_rgb1, _ = torch.max(im1, 1)
        max_rgb1 = max_rgb1.unsqueeze(1) 
        r_reg = self.L2(L1*R1, im1) + self.L2(R1, im1/(L1.detach() + 1e-8))
        l_reg = self.L2(L1, max_rgb1) + tv_loss(L1)

        rl_loss = 0
        rl_loss = r_reg + l_reg

        pl_loss = 0
        
        return c_loss, rl_loss, pl_loss

class LossStage2(nn.Module):
    def __init__(self, opt):
        super(LossStage2, self).__init__()
        self.L2 = torch.nn.MSELoss()
        self.L1 = torch.nn.L1Loss()
        self.edge = sobelFilter()
        self.lower = opt.lower         
        self.upper = opt.upper 
        self.beta1 = opt.beta1
        self.beta2 = opt.beta2
        self.mid   = 0.5
        self.opt   = opt
        
    def forward(self, L1, L2, 
                R1, R2, 
                R1_clean, R2_clean, dataset_flag,
                im1, im2):
        B, C, H, W = R1_clean.shape
        #r consistency
        c_loss = 0
        c_loss += self.L2(R1_clean, R2_clean) * self.opt.con_loss_w
        c_loss += self.L2(R1_clean, R1) * self.opt.recon_loss_w
        
        #r and l regularisation    (L1.detach() + 1e-8)
        max_rgb1, _ = torch.max(im1, 1)
        max_rgb1 = max_rgb1.unsqueeze(1) 
        r_reg = self.L2(L1*R1, im1) + self.L2(R1, im1/(L1.detach() + 1e-8))
        l_reg = self.L2(L1, max_rgb1) + tv_loss(L1)

        rl_loss = 0
        rl_loss = r_reg + l_reg

        pl_loss = 0
        dataset_weight = []
        for e in dataset_flag:
            if e == 'LOL':
                dataset_weight.append(1)
            else:
                dataset_weight.append(0)
        dataset_weight = torch.tensor(dataset_weight, dtype=float).cuda()#.unsqueeze(1).unsqueeze(1).unsqueeze(1)
        R1_clean_edge_x, R1_clean_edge_y = self.edge(R1_clean)
        weight_map_R1, _ = generate_Md(R1.detach(), beta1=self.beta1, beta2=self.beta2, mid=self.mid, lower=self.lower, upper=self.upper)
        weight_map_R1_clean, _ = generate_Md(R1_clean.detach(), beta1=self.beta1, beta2=self.beta2, mid=self.mid, lower=self.lower, upper=self.upper)
       
        flat_weight_map = norm(torch.abs(weight_map_R1-weight_map_R1_clean)).detach()
        text_map = 1-flat_map
        
        target = torch.zeros_like(R1_clean_edge_x)

        
        pl1_loss = 0
        pl1_loss += torch.abs(R1_clean_edge_x-target) * flat_weight_map
        pl1_loss += torch.abs(R1_clean_edge_y-target) * flat_weight_map
        pl1_loss = torch.mean(pl1_loss, dim=[1,2,3]) * dataset_weight
        pl1_loss = torch.mean(pl1_loss) * self.opt.flat_loss_w
        pl_loss = pl_loss  * dataset_weight

        pl2_loss = 0
        pl2_loss += torch.abs(R1_clean-R1.detach()) * text_map
        pl2_loss = torch.mean(pl2_loss, dim=[1,2,3])
        pl2_loss = torch.mean(pl2_loss) * self.opt.text_loss_w 
            
        pl_loss = pl1_loss + pl2_loss

        return c_loss, rl_loss, pl_loss
    
    
    
class LossSingle(nn.Module):
    def __init__(self, opt):
        super(LossSingle, self).__init__()
        self.L2 = torch.nn.MSELoss()
        self.L1 = torch.nn.L1Loss()
        self.edge = sobelFilter()
        self.lower = opt.lower         
        self.upper = opt.upper 
        self.beta1 = opt.beta1
        self.beta2 = opt.beta2
        self.mid   = 0.5
        self.opt   = opt
        
    def forward(self, L1, L2, 
                R1, R2, 
                R1_clean, R2_clean, dataset_flag,
                im1, im2):
        B, C, H, W = R1_clean.shape
        #r consistency
        c_loss = 0
        c_loss += self.L2(R1_clean, R2_clean) 
        c_loss += self.L2(R1_clean, R1)
        
        #r and l regularisation    (L1.detach() + 1e-8)
        max_rgb1, _ = torch.max(im1, 1)
        max_rgb1 = max_rgb1.unsqueeze(1) 
        r_reg = self.L2(L1*R1, im1) + self.L2(R1, im1/(L1.detach() + 1e-8))
        l_reg = self.L2(L1, max_rgb1) + tv_loss(L1)

        rl_loss = 0
        rl_loss = r_reg + l_reg

        pl_loss = 0
        dataset_weight = []
        for e in dataset_flag:
            if e == 'LOL':
                dataset_weight.append(1)
            else:
                dataset_weight.append(0)
        dataset_weight = torch.tensor(dataset_weight, dtype=float).cuda()#.unsqueeze(1).unsqueeze(1).unsqueeze(1)
        R1_clean_edge_x, R1_clean_edge_y = self.edge(R1_clean)
        weight_map_R1, _ = generate_Md(R1.detach(), beta1=self.beta1, beta2=self.beta2, mid=self.mid, lower=self.lower, upper=self.upper)
        weight_map_R1_clean, _ = generate_Md(R1_clean.detach(), beta1=self.beta1, beta2=self.beta2, mid=self.mid, lower=self.lower, upper=self.upper)
       
        flat_weight_map = norm(torch.abs(weight_map_R1-weight_map_R1_clean)).detach()
        text_map = 1-flat_map
        
        target = torch.zeros_like(R1_clean_edge_x)

        
        pl1_loss = 0
        pl1_loss += torch.abs(R1_clean_edge_x-target) * flat_weight_map
        pl1_loss += torch.abs(R1_clean_edge_y-target) * flat_weight_map
        pl1_loss = torch.mean(pl1_loss, dim=[1,2,3]) * dataset_weight
        pl1_loss = torch.mean(pl1_loss) * 0.1
        pl_loss = pl_loss  * dataset_weight

        pl2_loss = 0
        pl2_loss += torch.abs(R1_clean-R1.detach()) * text_map
        pl2_loss = torch.mean(pl2_loss, dim=[1,2,3])
        pl2_loss = torch.mean(pl2_loss) * 1    
            
        pl_loss = pl1_loss + pl2_loss

        return c_loss, rl_loss, pl_loss


def format_time(s):
    s = round(s)

    #s
    if s < 60:
        return f'{s}s'
    #m s
    elif s < 60 * 60:
        return f'{s // 60:02}m {s % 60:02}s'
    #h m s
    elif s < 60 * 60 * 24:
        return f'{s // (60*60):02}h {(s // 60) % 60 :02}m {s % 60:02}s'
    #d h m
    else:
        return f'{s // (24 * 60 * 60)}d {(s // (60 * 60)) % 24:02}h {(s // 60) % 60}m'

    
def gradient(img):
    height = img.size(2)
    width = img.size(3)
    gradient_h = (img[:,:,2:,:]-img[:,:,:height-2,:]).abs()
    gradient_w = (img[:, :, :, 2:] - img[:, :, :, :width-2]).abs()
    return gradient_h, gradient_w

def tv_loss(illumination):
    gradient_illu_h, gradient_illu_w = gradient(illumination)
    loss_h = gradient_illu_h
    loss_w = gradient_illu_w
    loss = loss_h.mean() + loss_w.mean()
    return loss

def C_loss(R1, R2):
    loss = torch.nn.MSELoss()(R1, R2) 
    return loss

def R_loss(L1, R1, im1, X1):
    max_rgb1, _ = torch.max(im1, 1)
    max_rgb1 = max_rgb1.unsqueeze(1) 
    loss1 = torch.nn.MSELoss()(L1*R1, X1) + torch.nn.MSELoss()(R1, X1/L1.detach())
    loss2 = torch.nn.MSELoss()(L1, max_rgb1) + tv_loss(L1)
    return loss1 + loss2

def P_loss(im1, X1):
    loss = torch.nn.MSELoss()(im1, X1)
    return loss