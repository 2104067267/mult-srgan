import argparse
from torch.utils.data import DataLoader
from utils import *
from model import Discriminator,NewGenerator,Generator
from pathlib import Path
import torch.nn as nn
from utils import *
from dataset import TestDatasetFromFolder
from model import NewGenerator
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


parser = argparse.ArgumentParser(description='Test Image')

parser.add_argument('--hr_image_path', default="D:\Pycharm\Py_projects\project2\BERT\SRGAN-master-1\SRGAN-master\data\DIV2K_train_HR\DIV2K_train_HR", type=str,help = "choose whether to output the evaluation results.")
parser.add_argument('--lr_image_path', default ="D:\Pycharm\Py_projects\project2\BERT\SRGAN-master-1\SRGAN-master\data\lr_image" ,type=str, help='test low resolution image name')
parser.add_argument('--produce_lr_images', default=False, type=bool, help='choose whether to generate low-resolution images.')
parser.add_argument('--model_path', default='D:\Pycharm\Py_projects\project2\BERT\SRGAN-master-1\SRGAN-master\pretrained_model', type=str, help='generator model epoch name')
parser.add_argument('--result_save_dirt', default="D:\Pycharm\Py_projects\project2\BERT\SRGAN-master-1\SRGAN-master//test_outputs", type=str)

parser.add_argument('--upscale_factor', default=2, type=int, choices=[1,2],help='super resolution upscale factor')


def test():
    opt            =  parser.parse_args()
    upscale_factor =  opt.upscale_factor
    hr_image_path  =  opt.hr_image_path
    lr_image_path  =  opt.lr_image_path
    produce_lr_images = opt.produce_lr_images
    model_path     =  opt.model_path
    result_save_dirt = opt.result_save_dirt

    #生成低分辨率图像
    if produce_lr_images:
        lr_image_produce(hr_image_path,hr_image_path,lr_image_path,upscale_factor)
        print("produce low-resolution images successfully")
        return

    #创建结果保存文件夹，以当前时间命名
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_folder = os.path.join(result_save_dirt ,current_time)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    save_image_folder = os.path.join(save_folder,'images')
    if not os.path.exists(save_image_folder):
        os.makedirs(save_image_folder)

    #数据集
    test_set = TestDatasetFromFolder(hr_image_path,lr_image_path,10)
    test_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1, shuffle=False)

    # 生成器
    netG = NewGenerator(upscale_factor)
    load_model(netG, model_path+"/generator", "generator")
    print('#generator parameters:', sum(param.numel() for param in netG.parameters()))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    netG.to(device).eval()

    results = {'psnr': [], 'ssim': [],'entropy':[]}
    to_pilimage = transforms.ToPILImage()

    with torch.no_grad():
        for k, (lr_img,hr_img) in enumerate(test_loader):
            lr_img = lr_img.to(device)
            hr_img = hr_img.to(device)
            fake_img = netG(lr_img)

            if hr_image_path is not None:
                psnr, ssim, entropy = evaluate_super_resolution(hr_img.detach().cpu().squeeze(0).permute(1, 2, 0).numpy(), fake_img.detach().cpu().squeeze(0).permute(1, 2, 0).numpy())
                results['psnr'].append(psnr)
                results['ssim'].append(ssim)
                results['entropy'].append(entropy)

            fake_img = to_pilimage(fake_img.squeeze(0).cpu())
            fake_img.save(save_image_folder+f"//{k}.png")

    if hr_image_path is not None:
        save_data_as_csv(save_folder,results,'evaluations')
        for key in results.keys():
            play_curves([key],[results[key]],1,1,'image',1,key,save_folder)


if __name__ == '__main__':
    import cv2

    video_path ="C://Users//21040\Downloads\无标题视频 (1).mp4" # 替换为实际的视频路径
    cap = cv2.VideoCapture(video_path)
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec = chr(fourcc & 0xFF) + chr((fourcc >> 8) & 0xFF) + chr((fourcc >> 16) & 0xFF) + chr((fourcc >> 24) & 0xFF)
    print("视频编码格式：", codec)

    # func()
    # test()

def func():
    import csv
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator

    def read_csv_file(file_path, column_name):
        data = []
        try:
            with open(file_path, 'r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    if column_name in row:
                        try:
                            value = float(row[column_name])
                            data.append(value)
                        except ValueError:
                            continue
        except FileNotFoundError:
            print(f"文件 {file_path} 未找到。")
        return {column_name: data}

    def play_curves(datas, locator_x, locator_y, label_x, label_y, name, save_path):
        color = ['red', 'blue', 'green', 'cyan', 'magenta']
        x = ['ours', 'real-esrgan', 'srgan']
        plt.figure(figsize=(10, 5))
        for k, y in enumerate(datas):
            data = y[name]
            avg = sum(data) / len(data)
            plt.plot(data, label=f'{x[k]} (avg: {avg:.4f})', color=color[k])
            ax = plt.gca()
            x_locator = MultipleLocator(locator_x)
            y_locator = MultipleLocator(locator_y)
            ax.xaxis.set_major_locator(x_locator)
            ax.yaxis.set_major_locator(y_locator)

        plt.title(name)
        plt.xlabel(label_x)
        plt.ylabel(label_y)
        plt.legend()
        plt.grid(True)

        # 保存图像到指定路径
        save_file = f"{save_path}/{name}.png"
        plt.savefig(save_file)

        plt.show()
        plt.close()

    name = 'psnr'
    r1 = read_csv_file(
        r"D:\Pycharm\Py_projects\project2\BERT\SRGAN-master-1\SRGAN-master\test_outputs\20250426_185403\evaluations.csv", name)
    r2 = read_csv_file(
        r"D:\Pycharm\Py_projects\project2\BERT\SRGAN-master-1\SRGAN-master\result\20250417_090635\losses.csv", name)
    # r3 = read_csv_file(
    #     r"D:\Pycharm\Py_projects\project2\BERT\SRGAN-master-1\SRGAN-master\result\20250417_094831\losses.csv", name)
    r4 = read_csv_file(
        r"D:\Pycharm\Py_projects\project2\BERT\SRGAN-master-1\SRGAN-master\result\20250417_104203\losses.csv", name)

    play_curves([r1, r2, r4], 1, 1, 'image', 'value', name,
                r"D:\Pycharm\Py_projects\project2\BERT\SRGAN-master-1\SRGAN-master\test_outputs")