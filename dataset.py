import os
import random
import torch
import torch.utils.data as data
import numpy as np
from os import listdir
from os.path import join
from PIL import Image, ImageOps
from random import randrange
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

import pandas as pd


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".bmp", ".JPG", ".jpeg"])

def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    return img

class Testset(data.Dataset):
    def __init__(self, data_dir, datasetname):
        self.lq_path = join(data_dir, 'low')
        self.hq_path = join(data_dir, 'high')
        self.name_list = [e for e in listdir(self.lq_path) if is_image_file(e)]
        self.datasetname = datasetname

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        lq_name = self.name_list[index] 

        if self.datasetname == 'LOLv2':
            hq_name = 'normal' + lq_name[3:]
        elif self.datasetname == 'LOLv2Syn':
            hq_name = lq_name
        elif self.datasetname == 'SICE':
            hq_name = lq_name.split('_')[0] + '.' + lq_name.split('.')[1]
        else:
            hq_name = lq_name

        lq_img = Image.open(join(self.lq_path, lq_name))
        hq_img = Image.open(join(self.hq_path, hq_name))
        
        lq_img = F.to_tensor(lq_img)
        hq_img = F.to_tensor(hq_img)

        return lq_img, hq_img, lq_name, hq_name
    
class Inferset(data.Dataset):
    def __init__(self, data_dir):
        self.lq_path = join(data_dir)
        self.name_list = [e for e in listdir(self.lq_path) if is_image_file(e)]
        self.transform = transforms.Compose([
                            # transforms.Resize(size=(512, 512)),
                            ])
        
    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        lq_name = self.name_list[index] 
        lq_img = Image.open(join(self.lq_path, lq_name))
        lq_img = F.to_tensor(lq_img)
        # lq_img = self.transform(lq_img)

        return lq_img, lq_name


