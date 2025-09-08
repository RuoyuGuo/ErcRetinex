import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import time
import argparse
from net.ercnet import net
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset import *

from skimage.metrics import peak_signal_noise_ratio as cal_psnr
from skimage.metrics import structural_similarity as cal_ssim
import numpy as np
from PIL import Image
import lpips
from tqdm import tqdm

parser = argparse.ArgumentParser(description='PairLIE')
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader to use')
parser.add_argument('--rgb_range', type=int, default=1, help='maximum value of RGB')

 
parser.add_argument('--model', type=str, required=True, help='Pretrained model')   
parser.add_argument('--data_path', type=str, required=True, help='testing datset path')
parser.add_argument('--output_path', type=str, required=True, help='Output image path')
parser.add_argument('--alpha', type=float, default=0.08, help='lighting strength')

opt = parser.parse_args() 

print('===> Loading datasets')
test_set = Inferset(opt.data_path)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)

print('===> Building model')
model = net().cuda()
model.load_state_dict(torch.load(opt.model, map_location=lambda storage, loc: storage))
print('Pre-trained model is loaded.')

os.makedirs(opt.output_path, exist_ok=True)
os.makedirs(opt.output_path + '/L/', exist_ok=True)
os.makedirs(opt.output_path + '/R/', exist_ok=True)
os.makedirs(opt.output_path + '/I/', exist_ok=True)  
os.makedirs(opt.output_path + '/R_clean/', exist_ok=True) 


def eval():
    torch.set_grad_enabled(False)
    model.eval()
    print('\nEvaluation:')
    total_psnr = 0
    total_ssim = 0
    total_num = 0
    
    with torch.no_grad():
        for batch in tqdm(testing_data_loader):
            input, name = batch[0], batch[1]

            L, R, R_clean = model(input.cuda())

            I = torch.pow(L, opt.alpha) * R_clean

            L = L.cpu()
            R = R.cpu()
            I = I.cpu()
            R_clean = R_clean.cpu()     

            L_img = transforms.ToPILImage()(L.squeeze(0))
            R_img = transforms.ToPILImage()(R.squeeze(0))              
            R_clean_img = transforms.ToPILImage()(R_clean.squeeze(0))  
            I_img = transforms.ToPILImage()(I.squeeze(0))  

            save_name = name[0].split('.')
            save_name = save_name[0] + '.png'
            
            L_img.save(opt.output_path + '/L/' + save_name)
            R_img.save(opt.output_path + '/R/' + save_name)
            R_clean_img.save(opt.output_path + '/R_clean/' + save_name)  
            I_img.save(opt.output_path + '/I/' + save_name)   


    torch.set_grad_enabled(True)

    
eval()


