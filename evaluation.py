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
from noise_est import *

parser = argparse.ArgumentParser(description='Erc')
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader to use')
parser.add_argument('--rgb_range', type=int, default=1, help='maximum value of RGB')


parser.add_argument('--model', type=str, required=True, help='Pretrained model')   
parser.add_argument('--output_path', type=str, required=True, default='./results/lol', help='Output image path')
parser.add_argument('--data_test', type=str, required=True, help='Selecting your dataset, [LOLv1, LOLv2, SICE, LOLv2Syn]')
parser.add_argument('--data_path', type=str, required=True, help='testing datset path') 
opt = parser.parse_args()


print('===> Loading datasets')
test_set = Testset(opt.data_path, opt.data_test)
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

def t(img):
    def to_4d(img):
        assert len(img.shape) == 3
        assert img.dtype == np.uint8
        img_new = np.expand_dims(img, axis=0)
        assert len(img_new.shape) == 4
        return img_new

    def to_CHW(img):
        return np.transpose(img, [2, 0, 1])

    def to_tensor(img):
        return torch.Tensor(img)

    return to_tensor(to_4d(to_CHW(img))) / 127.5 - 1

def calculate_lpips_lol(img, img2, device, model):
    tA = t(img).to(device)
    tB = t(img2).to(device)
    dist01 = model.forward(tA, tB).item()
    return dist01

lpmodel = lpips.LPIPS(net='alex').cuda()

def eval():
    torch.set_grad_enabled(False)
    model.eval()
    print('\nEvaluation:')
    total_psnr = 0
    total_ssim = 0
    total_lpips = 0
    total_noise_level = 0
    total_num = 0
    
    with torch.no_grad():
        for batch in tqdm(testing_data_loader):
            input, target, name, name2 = batch[0], batch[1], batch[2], batch[3]

            L, R, R_clean = model(input.cuda())

            if opt.data_test == 'LOLv1':
                I = torch.pow(L, 0.08) * R_clean  #0.08
            elif opt.data_test == 'LOLv2':
                I = torch.pow(L, 0.16) * R_clean  
            elif opt.data_test == 'LOLv2Syn':
                I = torch.pow(L, 0.18) * R_clean
            elif opt.data_test == 'SICE':         #0.20
                I = torch.pow(L, 0.18) * R_clean
            elif opt.data_test == 'infer':
                I = torch.pow(L, 0.08) * R_clean

            L = L.cpu()
            R = R.cpu()
            I = I.cpu()
            R_clean = R_clean.cpu()     

            L_img = transforms.ToPILImage()(L.squeeze(0))
            R_img = transforms.ToPILImage()(R.squeeze(0))
            I_img = transforms.ToPILImage()(I.squeeze(0))                
            R_clean_img = transforms.ToPILImage()(R_clean.squeeze(0))  
            target = transforms.ToPILImage()(target.squeeze(0))

            target = np.array(target)

            temp_psnr = cal_psnr(np.array(I_img), target, data_range=255)
            temp_ssim = cal_ssim(np.array(I_img), target, data_range=255, channel_axis=2)
            temp_lpips = calculate_lpips_lol(np.array(I_img), target, 'cuda', lpmodel)
            temp_noise_level = noise_estimate(np.array(I_img))
            total_psnr += temp_psnr
            total_ssim += temp_ssim
            total_lpips += temp_lpips
            total_noise_level += temp_noise_level
            total_num += 1

            # L_img.save(opt.output_path + '/L/' + name[0])
            # R_img.save(opt.output_path + '/R/' + name[0])
            I_img.save(opt.output_path + '/I/' + name[0])  
            # R_clean_img.save(opt.output_path + '/R_clean/' + name[0])                        
        
        
    torch.set_grad_enabled(True)
    
    print(f'PSNR: {total_psnr/total_num:.4f}')
    print(f'SSIM: {total_ssim/total_num:.4f}')
    print(f'lpips: {total_lpips/total_num:.4f}')
    print(f'noise: {total_noise_level/total_num:.4f}')
    
eval()


