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


class TrainingDataset(data.Dataset):
    def __init__(self, data_dir, crop_size):
        self.data_dir = data_dir
        self.crop_size = crop_size
        self.data_name =  listdir(self.data_dir)

    def transform(self, imgs):
        # Random crop
        i, j, h, w = transforms.RandomCrop.get_params(
            imgs[0], output_size=(self.crop_size, self.crop_size))
        
        for idx, e in enumerate(imgs):
            imgs[idx] = F.crop(e, i, j, h, w)

        rotate_degree = random.randint(0,3) * 90
        for idx, e in enumerate(imgs):
            imgs[idx] = F.rotate(e, rotate_degree)

        # Random horizontal flipping
        if random.random() > 0.5:
            for idx, e in enumerate(imgs):
                imgs[idx] = F.hflip(e)

        # Random vertical flipping
        if random.random() > 0.5:
            for idx, e in enumerate(imgs):
                imgs[idx] = F.vflip(e)

        # Transform to tensor
        for idx, e in enumerate(imgs):
            imgs[idx] = F.to_tensor(e)
        return imgs

    def __len__(self):
        return len(listdir(self.data_dir))
        
    def __getitem__(self, index):
        # img_path = join(self.data_dir, str(index+1))
        img_path = join(self.data_dir, self.data_name[index])
        img_list = [e for e in listdir(img_path) if is_image_file(e)]
        num = len(img_list)
        index1 = random.randint(1,num)
        index2 = random.randint(1,num)
        while abs(index1 - index2) == 0:
            index2 = random.randint(1,num)

        im1 = load_img(join(img_path, img_list[index1-1]))
        im2 = load_img(join(img_path, img_list[index2-1]))

        # #SICE data

        
        file1 = img_list[index1-1]
        file2 = img_list[index2-1]

        if '.JPG' in file1:
            dataset_flag = 'SICE'
        # #LOL data
        else:
            dataset_flag = 'LOL'

        im1, im2 = self.transform([im1, im2])

        data = {'im1': im1,
                'im2': im2, 
                'file1': file1,
                'file2': file2,
                'dataset_flag': dataset_flag,}
        
        return data 