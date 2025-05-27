
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize, Normalize
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import torch.nn.init as init
import torch.nn as nn
import torch

#判断是否是图像数据
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])

#计算裁剪大小
def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)

#训练高分辨率数据集 图像处理函数
def train_hr_transform(crop_size):
    return Compose([
        CenterCrop(crop_size),
        ToTensor(),
    ])

#训练低分辨率图像处理
def train_lr_BICUBIC(crop_size, upscale_factor):
    return Compose([
        ToPILImage(),
        Resize(crop_size // upscale_factor, interpolation=Image.BICUBIC),#双三次线性插值法
        ToTensor()
    ])

def train_lr_gaussian(kernel_size = 3, sigma = 0.5):
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.GaussianBlur(kernel_size=kernel_size , sigma=sigma),
        transforms.ToTensor()
    ])

#
def display_transform():
    return Compose([
        ToPILImage(),
        Resize(400),
        CenterCrop(400),
        ToTensor()
    ])


def display_image(img):
    num_images = len(img)
    fig, axs = plt.subplots(1, num_images, figsize=(35,35))
    for i,image in enumerate(img):
        x = image.permute(1,2,0).detach().cpu().numpy()
        axs[i].imshow(x)
    plt.show()
    plt.close()
    return


def calculate_entropy(image):
    if len(image.shape) == 3:
        # 如果是多通道图像，转换为灰度图
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([image], [0], None, [256], [0, 1])
    hist = hist.ravel() / hist.sum()
    entropy = -np.sum([p * np.log2(p) for p in hist if p > 0])
    return entropy


def evaluate_super_resolution(original_image, super_resolved_image):
    # 计算PSNR:峰值信噪比：衡量像素差异：先计算像素的均方误差，在转化到分贝空间：10Xlog10()
    psnr_value = psnr(original_image, super_resolved_image, data_range=255)
    # 计算SSIM:结构相似性指数：从亮度、对比度、结构纹理 三个维度评估图像
    # 对比度比较：计算俩图像的 像素的方差 和协方差，评估对比度度一致性、结构比较，综合指数
    # 返回综合指数为 三者的乘积 【0，1】，越接近1越相似
    if len(original_image.shape) == 2:
        # 单通道图像
        ssim_value = ssim(original_image, super_resolved_image, data_range=255)
    else:
        # 多通道图像
        ssim_value = ssim(original_image, super_resolved_image, data_range=255, multichannel=True,channel_axis = 2,win_size=7)
    # 计算信息熵
    entropy = calculate_entropy(super_resolved_image) #衡量细节、信息丰富程度
    return psnr_value, ssim_value, entropy


import re
import os
pattern = r'[._]'


def save_model(model,save_path,save_name):
    if not save_path.exists():
        os.makedirs(save_path)
    i = -1
    for filename in os.listdir(save_path):
        name,s,_ = filename.split('.')
        if name == save_name : i = max(i,int(s))
    i+=1
    torch.save(model.state_dict(),os.path.join(save_path, f"{save_name}.{i}.pth"))


def load_model(model,dir_root,model_name,i=None):
    if i is None:
        if os.path.exists(dir_root):
            i = -1
            for filename in os.listdir(dir_root):
                name,s,_= filename.split('.')
                if name == model_name : i = max(i,int(s))
    if i>=0 :model.load_state_dict(torch.load(os.path.join(dir_root, f"{model_name}.{i}.pth")))



import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from os import listdir, makedirs, path


def motion_blur(image, degree, angle):
    # 确保image是NumPy数组
    if not isinstance(image, np.ndarray):
        image = np.array(image)
    # 确保image的数据类型为uint8
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
    motion_blur_kernel = np.diag(np.ones(degree))
    motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))
    motion_blur_kernel = motion_blur_kernel / degree
    blurred = cv2.filter2D(image, -1, motion_blur_kernel)
    return Image.fromarray(blurred)


def defocus_blur(image, radius):
    # 确保image是NumPy数组
    if not isinstance(image, np.ndarray):
        image = np.array(image)
    # 确保image的数据类型为uint8
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    kernel = cv2.getGaussianKernel(2 * radius + 1, 0)
    kernel2D = np.outer(kernel, kernel.transpose())
    blurred = cv2.filter2D(image, -1, kernel2D)
    return Image.fromarray(blurred)

def single_lr_image_produce(hr_image_path, lr_image_save_path):
    transform = transforms.Compose([
        transforms.GaussianBlur(kernel_size=3, sigma=3),
    ])
    image = Image.open(hr_image_path)
    image = transform(image)
    image = motion_blur(image, 1, 0)
    image = defocus_blur(image, 9)
    image.save(path.join(lr_image_save_path,"0000.png"))

from torchvision.transforms import InterpolationMode

def lr_image_produce(hr_image_dirt, hr_save_path,lr_save_path,up_scale):
    gaussionblur = transforms.Compose([
        transforms.GaussianBlur(kernel_size=5, sigma=5),
    ])
    for image_name in listdir(hr_image_dirt)[0:20]:
        image = Image.open(path.join(hr_image_dirt, image_name))
        w,h = image.size
        if w%up_scale != 0 or h%up_scale != 0:
            w -= w%up_scale
            h -= h%up_scale
            image = CenterCrop((h,w))(image)
        lr_image = defocus_blur(image, 5)
        lr_image = gaussionblur(lr_image)
        lr_image = Resize((h//up_scale,w//up_scale), interpolation=InterpolationMode.BILINEAR)(lr_image)
        lr_image.save(os.path.join(lr_save_path, image_name))

def lr_image_produce1(hr_image_dir, lr_image_save_path, hr_image_save_path, up_scale):
    # 定义低分辨率处理流程（包含模糊和缩放）
    blur_transform = transforms.Compose([
        transforms.GaussianBlur(kernel_size=9, sigma=8),
    ])

    for image_name in listdir(hr_image_dir)[:100]:
        try:
            with Image.open(os.path.join(hr_image_dir,image_name)) as img:
                hr_image = img
                hr_width, hr_height = hr_image.size
                sub_hr_height = hr_height // 2  # 子图高度
                sub_hr_width = hr_width // 2  # 子图宽度

                resize_transform = transforms.Resize(
                    (sub_hr_height // up_scale, sub_hr_width // up_scale),
                    interpolation=InterpolationMode.BILINEAR
                )

                # 划分区域坐标（4个子图：左上、右上、左下、右下）
                crop_regions = [
                    (0, 0, sub_hr_width, sub_hr_height),  # 左上 (x1,y1,x2,y2)
                    (sub_hr_width, 0, hr_width, sub_hr_height),  # 右上
                    (0, sub_hr_height, sub_hr_width, hr_height),  # 左下
                    (sub_hr_width, sub_hr_height, hr_width, hr_height)  # 右下
                ]

                for idx, (x1, y1, x2, y2) in enumerate(crop_regions, 1):
                    # 1. 裁剪高分辨率子图
                    sub_hr_image = hr_image.crop((x1, y1, x2, y2))
                    lr_image = sub_hr_image

                    # # 2. 生成低分辨率子图（应用模糊和缩放）
                    # #lr_image = motion_blur(sub_hr_image, 1, 0)
                    # lr_image = resize_transform(sub_hr_image)
                    # lr_image = defocus_blur(lr_image, 5)
                    # lr_image = blur_transform(lr_image)

                    # 3. 生成文件名（添加子图编号后缀）
                    base_name, ext = path.splitext(image_name)
                    sub_hr_name = f"{base_name}_part{idx}{ext}"
                    sub_lr_name = f"{base_name}_part{idx}_lr{ext}"

                    # 4. 保存图像
                    sub_hr_path = os.path.join(hr_image_save_path, sub_hr_name)
                    sub_lr_path = os.path.join(lr_image_save_path, sub_lr_name)

                    # sub_hr_image.save(sub_hr_path)
                    lr_image.save(sub_lr_path)

        except Exception as e:
            print(f"处理图像 {image_name} 时出错: {str(e)}")

import pandas as pd


def read_csv_file(file_path,name):
    try:
        df = pd.read_csv(file_path)
        results = {name:[]}
        for index, row in df.iterrows():
            results[name].append(row[name])
        return results
    except FileNotFoundError:
        print(f"错误：未找到文件 {file_path}。")
        return None
    except KeyError:
        print("错误：CSV 文件中缺少必要的列。")
        return None
    except Exception as e:
        print(f"错误：读取文件时出现未知错误 {e}。")
        return None


#滤波器可视化解释法
def normalize(image):
    return (image - image.min()) / (image.max() - image.min())
layer_activations = None
#查看某一层的输出，进行可视化

def filter_explanation(image, model,iteration=100, lr=1):
    # 检查 CUDA 是否可用
    x = torch.empty_like(image)
    # 深拷贝
    x.copy_(image)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = x.to(device)
    model.eval()
    layer_activations = None
    def hook(model, input, output):
        nonlocal layer_activations
        layer_activations = output
    hook_handle = model.net[8].register_forward_hook(hook)
    try:
        x.requires_grad_()
        # 前向传播
        model(x)
        avg_activations = layer_activations.detach().cpu()
        optimizer = torch.optim.Adam([x], lr=lr)

        for iter in range(iteration):
            optimizer.zero_grad()
            predict = model(x)
            objective = predict
            objective.backward()
            optimizer.step()

        filter_visualizations = x.detach().cpu()

        num_images = x.size(0) if len(x.size()) > 0 else 1
        fig, axs = plt.subplots(3, num_images, figsize=(12, 12))
        if num_images == 1:
            axs = axs.reshape(3, 1)
        # 绘制原始图像
        image = image.detach().cpu()
        for i,img in enumerate(image):
            axs[0][i].imshow(img.permute(1,2,0))

        for i,img in enumerate(avg_activations):
            axs[1][i].imshow(normalize(img[0,:,:]))

        for i,img in enumerate(filter_visualizations):
            axs[2][i].imshow(normalize(img.permute(1,2,0)))

        plt.show()
        plt.close()
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # 确保钩子被移除
        hook_handle.remove()


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight)
            if m.bias is not None:
                init.zeros_(m.bias)





def plot_psnr_ssim(models, psnr_values, ssim_values):
    # 设置图形大小
    plt.figure(figsize=(12, 6))

    # 绘制 PSNR 柱状图
    bar_width = 0.35
    index = np.arange(len(models))
    psnr_bars = plt.bar(index, psnr_values, bar_width, label='PSNR')

    # 添加 PSNR 数据标签
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            plt.annotate(f'{height:.2f}',
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3),  # 3 points vertical offset
                         textcoords="offset points",
                         ha='center', va='bottom')

    add_labels(psnr_bars)

    # 绘制 SSIM 折线图
    ssim_line = plt.plot(index, ssim_values, marker='o', label='SSIM')

    # 添加 SSIM 数据标签
    for i, v in enumerate(ssim_values):
        plt.text(i, v, f'{v:.2f}', ha='center', va='bottom')

    # 设置图表标题和坐标轴标签
    plt.title('PSNR and SSIM Comparison of Different Models')
    plt.xlabel('Models')
    plt.ylabel('Values')
    plt.xticks(index, models)
    plt.legend()

    # 显示图形
    plt.tight_layout()
    plt.show()


def plot_multiple_ssim(models, ssim_data):
    # 样本数量
    num_samples = len(ssim_data[0])
    # 样本索引
    sample_index = np.arange(num_samples)

    # 设置图形大小
    plt.figure(figsize=(12, 6))

    # 绘制不同模型的 SSIM 折线图
    colors = ['r', 'g', 'b', 'm']  # 定义不同颜色
    for i, ssim_values in enumerate(ssim_data):
        plt.plot(sample_index, ssim_values, marker='o', color=colors[i], label=models[i])
        # 添加数据标签
        for j, v in enumerate(ssim_values):
            plt.text(j, v, f'{v:.2f}', ha='center', va='bottom')

    # 设置图表标题和坐标轴标签
    plt.title('SSIM Comparison of Different Models for Different Samples')
    plt.xlabel('Samples')
    plt.ylabel('SSIM Values')
    plt.xticks(sample_index)
    plt.legend()

    # 显示图形
    plt.tight_layout()
    plt.show()


import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

def plot_loss_curves(results):
    g_losses = results['g_loss']

    # 创建一个新的图形
    plt.figure(figsize=(10, 5))

    # 绘制生成器损失曲线
    plt.plot(g_losses, label='Generator Loss', color='red')
    # 设置横纵坐标的刻度间隔和显示格式
    ax = plt.gca()
    # 这里可以根据数据情况调整刻度间隔，例如假设 x 轴以 1 为间隔，y 轴以 0.001 为间隔
    x_locator = MultipleLocator(1)
    y_locator = MultipleLocator(1)
    ax.xaxis.set_major_locator(x_locator)
    ax.yaxis.set_major_locator(y_locator)

    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))


    # 设置图表标题和坐标轴标签
    plt.title('Generator and Discriminator Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    # 显示图例
    plt.legend()

    # 显示网格线
    plt.grid(True)
    plt.show()
    # 关闭图表
    plt.close()


import csv
import os
from datetime import datetime


def save_data_as_csv(save_folder,datas,name):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    csv_file_path = os.path.join(save_folder, name+'.csv')

    with open(csv_file_path, 'w', newline='') as csvfile:
        keys = list(datas.keys())
        writer = csv.DictWriter(csvfile, fieldnames=keys)#创建csv字典写入器
        writer.writeheader()#写入表头
        for i in range(len(datas[keys[0]])):
            Dict = {}
            for k in keys:
                Dict[k] = datas[k][i]
            writer.writerow(Dict)#写入每一行

    print(f"损失数据已保存到 {csv_file_path}")


def play_curves(keys, datas, locator_x, locator_y, label_x, label_y, name, save_path):
    color = ['red', 'blue', 'green', 'cyan', 'magenta']
    plt.figure(figsize=(10, 5))
    for k, data in enumerate(datas):
        avg = sum(data) / len(data)
        plt.plot(data, label=f'{keys[k]} (avg: {avg:.4f})', color=color[k])
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
    plt.savefig(save_path+"/"+name+'.png')

