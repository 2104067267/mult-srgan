import argparse
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import *
from loss import GeneratorLoss
from model import  Discriminator,NewGenerator
from pathlib import Path

from dataset import TrainDatasetFromFolder

parser = argparse.ArgumentParser(description='Train Super Resolution Models')

parser.add_argument('--hr_image_path', default="./data\DIV2K_train_HR\DIV2K_train_HR", type=str)
parser.add_argument('--model_path', default="./SRGAN-master\pretrained_model3", type=str)
parser.add_argument('--result_save_path', default="./SRGAN-master//train_outputs", type=str)
parser.add_argument('--upscale_factor', default=2, type=int, choices=[2],help='super resolution upscale factor')
parser.add_argument('--crop_size', default=256, type=int, help='training images crop size')
parser.add_argument('--batch_size', default=5, type=int)
parser.add_argument('--num_epochs', default=1, type=int, help='train epoch number')
parser.add_argument('--lr_rate', default=0.0001, type=int)

def train():
    opt            =  parser.parse_args()
    crop_size      =  opt.crop_size
    upscale_factor =  opt.upscale_factor
    num_epoch      =  opt.num_epochs
    hr_image_path  =  opt.hr_image_path
    model_path     =  opt.model_path
    result_save_path = opt.result_save_path
    lr_rate        =  opt.lr_rate
    batch_size     =  opt.batch_size

    #创建结果保存文件夹，以当前时间命名
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_folder = os.path.join(result_save_path ,current_time)
    os.makedirs(save_folder)

    # 训练数据集
    train_set = TrainDatasetFromFolder(hr_image_path,crop_size,upscale_factor,l=-1)
    train_loader = DataLoader(dataset=train_set, num_workers=1, batch_size=batch_size, shuffle=False)

    # 生成器 判别器 损失网络
    netG = NewGenerator(scale_factor=upscale_factor)
    netD = Discriminator(crop_size)
    generator_criterion = GeneratorLoss()
    print('# generator parameters:', sum(param.numel() for param in netG.parameters()))
    print('# discriminator parameters:', sum(param.numel() for param in netD.parameters()))

    load_model(netG, model_path+"\generator", "generator")
    load_model(netD, model_path+"\discriminator", "discriminator")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    netG.to(device)
    netD.to(device)
    generator_criterion.to(device)

    optimizerG = optim.Adam(netG.parameters(), lr=lr_rate)
    optimizerD = optim.Adam(netD.parameters(), lr=lr_rate)

    results = {'g_loss':[],'d_loss':[],'g_score':[],'d_score':[]}

    for epoch in range(1, num_epoch + 1):
        train_bar = tqdm(train_loader)
        running_results = {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0, 'd_score': 0, 'g_score': 0}
        netG.train()
        netD.train()

        for k, (lr_img, hr_img) in enumerate(train_bar):
            batch_size = lr_img.size(0)
            running_results['batch_sizes'] += batch_size

            lr_img = lr_img.to(device)
            hr_img = hr_img.to(device)

            fake_img = netG(lr_img)
            fake_out = netD(fake_img).mean()

            # 更新生成器
            optimizerG.zero_grad()
            g_loss = generator_criterion(fake_out, fake_img, hr_img)
            g_loss.backward()
            optimizerG.step()

            hr_out = netD(hr_img).mean()
            fake_out = netD(fake_img.detach()).mean()
            d_loss = 1 - hr_out + fake_out
            optimizerD.zero_grad()
            d_loss.backward()
            optimizerD.step()

            running_results['g_loss'] += g_loss.item() * batch_size
            running_results['d_loss'] += d_loss.item() * batch_size
            running_results['d_score'] += hr_out.item() * batch_size
            running_results['g_score'] += fake_out.item() * batch_size

            train_bar.set_description(desc='[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f' % (
                epoch, num_epoch, running_results['d_loss'] / running_results['batch_sizes'],
                running_results['g_loss'] / running_results['batch_sizes'],
                running_results['d_score'] / running_results['batch_sizes'],
                running_results['g_score'] / running_results['batch_sizes'])
            )

        results['g_loss'].append(running_results['g_loss'] / running_results['batch_sizes'])
        results['d_loss'].append(running_results['d_loss'] / running_results['batch_sizes'])
        results['d_score'].append(running_results['d_score'] / running_results['batch_sizes'])
        results['g_score'].append(running_results['g_score'] / running_results['batch_sizes'])

    #于csv文件格式 保存训练数据
    save_data_as_csv(save_folder, results,'loss')

    save_model(netG, model_path+"\generator", "generator")
    save_model(netD, model_path+"\discriminator", "discriminator")


if __name__ == '__main__':
    torch.cuda.empty_cache()
    train()

