import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import argparse
import random
import shutil
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.optim.lr_scheduler as lrs
from torch.utils.data import DataLoader
from net.ercnet import net

import time
from collections import OrderedDict

from dataset import TrainingDataset
from data import get_training_set, get_eval_set
from utils import *
from PIL import Image
import numpy as np
from torchvision import transforms

# Training settings
parser = argparse.ArgumentParser(description='PairLIE')
parser.add_argument('--batchSize', type=int, default=4, help='training batch size')
parser.add_argument('--nEpochs', type=int, default=400, help='number of epochs to train for')
parser.add_argument('--snapshots', type=int, default=20, help='Snapshots')
parser.add_argument('--start_iter', type=int, default=1, help='Starting Epoch')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate. Default=1e-4')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader to use')
parser.add_argument('--decay', type=int, default='100', help='learning rate decay type')
parser.add_argument('--gamma', type=float, default=0.5, help='learning rate decay factor for step decay')
parser.add_argument('--seed', type=int, default=123456789, help='random seed to use. Default=123')
parser.add_argument('--data_train', type=str, default='./dataset/')
parser.add_argument('--rgb_range', type=int, default=1, help='maximum value of RGB')
parser.add_argument('--save_folder', default='weights/', help='Location to save checkpoint models')
parser.add_argument('--output_folder', default='./trainingresult/', help='Location to save images')

parser.add_argument('--crop_size', type=int, default=128, help='crop image to to train')
parser.add_argument('--resume', type=int, default=0, help='load pretrained network')

parser.add_argument('--con_loss_w'  , type=float, default=1.0)
parser.add_argument('--flat_loss_w'  , type=float, default=0.1)
parser.add_argument('--text_loss_w'  , type=float, default=1.0)
parser.add_argument('--recon_loss_w'  , type=float, default=1.0)
parser.add_argument('--lower'  , type=float, default=0.2)
parser.add_argument('--upper'  , type=float, default=0.8)
parser.add_argument('--beta1'  , type=float, default=5.0)
parser.add_argument('--beta2'  , type=float, default=5.0)

opt = parser.parse_args()

def seed_torch(seed=opt.seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
seed_torch()
cudnn.benchmark = True

def train():
    model.train()
    loss_print = 0
    SINCE = time.time()
    for iteration, batch in enumerate(training_data_loader, 1):

        im1 = batch['im1'].cuda()
        im2 = batch['im2'].cuda()
        file1, file2 = batch['file1'], batch['file2']
        dataset_flag = batch['dataset_flag']

        L1, R1, R1_clean = model(im1)
        L2, R2, R2_clean = model(im2) 
        
        c_loss, rl_loss, pl_loss = criterias(L1, L2, 
                                            R1, R2,
                                            R1_clean, R2_clean, 
                                            dataset_flag,
                                            im1, im2,)

        loss =  c_loss * 1 + rl_loss * 1 + pl_loss * 1

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if iteration % 10 == 0:
            s = f"===> Epoch[{epoch}]({iteration}/{len(training_data_loader)}): "
            s +=  f"Loss: {c_loss.item():.4f} {rl_loss.item():.4f} {pl_loss:.4f} "
            s +=  f"Learning rate: lr={optimizer.param_groups[0]['lr']}."
            print(s)

    epoch_training_time = format_time(time.time()-SINCE)
    s = f'===>Training time: {epoch_training_time} <==='
    print(s)
    print()

    L = L1[[0]].cpu().detach()
    R = R1[[0]].cpu().detach()
    R_clean = R1_clean[[0]].cpu().detach()   
        
    L_img        = transforms.ToPILImage()(L.squeeze(0))
    R_img        = transforms.ToPILImage()(R.squeeze(0))
    R_clean_img  = transforms.ToPILImage()(R_clean.squeeze(0))  

    L_img.save(opt.output_folder + '/L/' + file1[0])
    R_img.save(opt.output_folder + '/R/' + file1[0])
    R_clean_img.save(opt.output_folder + '/R_clean/' + file1[0])  


def checkpoint(epoch):
    model_out_path = opt.save_folder+"epoch_{}.pth".format(epoch)
    torch.save(model.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

cuda = opt.gpu_mode
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

print('===> Loading datasets')
# train_set = get_training_set(opt.data_train, opt.crop_size)
train_set = TrainingDataset(opt.data_train, opt.crop_size)
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)

print('===> Building model ')
model = net().cuda()
        
        
if opt.resume != 0:
    resume_model_path = f'./weights/epoch_{opt.resume}.pth'
    model.load_state_dict(torch.load(resume_model_path, map_location=lambda storage, loc: storage))
    print('===> Load pretrained model ')
optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8)
criterias = LossSingle(opt).cuda()

milestones = []
for i in range(1, opt.nEpochs+1):
    if i % opt.decay == 0:
        milestones.append(i)

scheduler = lrs.MultiStepLR(optimizer, milestones, opt.gamma)

score_best = 0
# shutil.rmtree(opt.save_folder)
# os.mkdir(opt.save_folder)
for epoch in range(opt.start_iter, opt.nEpochs + 1):
    train()
    scheduler.step()
    if epoch % opt.snapshots == 0:
        checkpoint(epoch)          
        
