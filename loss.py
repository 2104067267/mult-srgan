import torch
from torch import nn
from torchvision.models.vgg import vgg16
from torch.nn import functional as F
import cv2

#颜色直方图损失：
def color_histogram_loss(gen_img, real_img):
        gen_hist = torch.histc(gen_img, bins=256, min=0, max=255)
        real_hist = torch.histc(real_img, bins=256, min=0, max=255)
        return F.mse_loss(gen_hist, real_hist)


def brightness_loss(gen_gray, real_gray):
    gen_hist = torch.histc(gen_gray, bins=256, min=0, max=255)
    real_hist = torch.histc(real_gray, bins=256, min=0, max=255)
    return F.mse_loss(gen_hist, real_hist)

def tensor_to_gray(image):
    r,g,b = image[0,:,:], image[1,:,:], image[2,:,:]
    return r*0.299+g*0.587+b*0.114

def Contrast_Loss(x, y):
    def local_contrast(img):
        local_mean = nn.functional.avg_pool2d(img, kernel_size=3, stride=1, padding=1)
        local_std = torch.sqrt(nn.functional.avg_pool2d((img - local_mean) ** 2, kernel_size=3, stride=1, padding=1))
        return local_std

    x_contrast = local_contrast(x)
    y_contrast = local_contrast(y)
    loss = nn.functional.mse_loss(x_contrast, y_contrast)
    return loss

def rgb_to_ycbcr(rgb):
    r = rgb[:, 0, :, :].unsqueeze(1)
    g = rgb[:, 1, :, :].unsqueeze(1)
    b = rgb[:, 2, :, :].unsqueeze(1)

    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.1687 * r - 0.3313 * g + 0.5 * b + 128
    cr = 0.5 * r - 0.4187 * g - 0.0813 * b + 128

    ycbcr = torch.cat([y, cb, cr], dim=1)
    return ycbcr

def spatial_color_brightness_loss(real_images, fake_images, patch_size=8):

    real_ycbcr = rgb_to_ycbcr(real_images)
    fake_ycbcr = rgb_to_ycbcr(fake_images)

    real_brightness = real_ycbcr[:, 0:1, :, :]
    fake_brightness = fake_ycbcr[:, 0:1, :, :]

    real_chroma = real_ycbcr[:, 1:, :, :]
    fake_chroma = fake_ycbcr[:, 1:, :, :]

    batch_size, channels, height, width = real_chroma.size()

    # 计算局部区域的数量
    num_patches_h = height // patch_size
    num_patches_w = width // patch_size

    color_loss = 0
    brightness_loss = 0
    criterion = nn.MSELoss()

    # 遍历每个局部区域
    for i in range(num_patches_h):
        for j in range(num_patches_w):
            # 提取局部区域的色度
            real_chroma_patch = real_chroma[:, :, i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size]
            fake_chroma_patch = fake_chroma[:, :, i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size]

            # 计算局部区域的颜色损失
            local_color_loss = criterion(fake_chroma_patch, real_chroma_patch)
            color_loss += local_color_loss

            # 提取局部区域的亮度
            real_brightness_patch = real_brightness[:, :, i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size]
            fake_brightness_patch = fake_brightness[:, :, i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size]

            # 计算局部区域的亮度损失
            local_brightness_loss = criterion(fake_brightness_patch, real_brightness_patch)
            brightness_loss += local_brightness_loss

    # 平均颜色损失
    color_loss /= (num_patches_h * num_patches_w)
    # 平均亮度损失
    brightness_loss /= (num_patches_h * num_patches_w)

    # 合并颜色和亮度损失
    total_loss = 2e-4*color_loss + 2e-4*brightness_loss

    return total_loss

class GeneratorLoss(nn.Module):
    def __init__(self):
        super(GeneratorLoss, self).__init__()

        vgg = vgg16(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()

        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()
        self.tv_loss = TVLoss()


    def forward(self, out_labels, out_images, target_images):
        #对抗损失：
        adversarial_loss = torch.mean(1 - out_labels)#对抗损失

        #感知损失：关注深层特征，纹理差异： 比较目标图像和生成图像 在高层特征上的差异 来判断相似性
        perception_loss = self.mse_loss(self.loss_network(out_images), self.loss_network(target_images))

        #图像损失：关注像素差异
        image_loss = self.mse_loss(out_images, target_images)
        #总变分损失： 衡量图像平滑度的损失函数
        #计算图像 在水平方向 和 垂直方向上 梯度变化差异，值越小说明：噪声小，值越大：说明噪声大，细节多
        tv_loss = self.tv_loss(out_images)

        # contrast_loss = Contrast_Loss(out_images,target_images)+ 2e-5 * contrast_loss
        bright_color_loss = spatial_color_brightness_loss(out_images, target_images)

        return 2*image_loss + 0.01 * adversarial_loss + 0.05 * perception_loss+ 2e-7 * tv_loss + bright_color_loss


#总变分损失模块：== 梯度幅值
class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :]) #垂直方向的像素数量
        count_w = self.tensor_size(x[:, :, :, 1:]) #水平方向的像素数量
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum() #计算垂直放向的差值
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum() #计算水平方向
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size
    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]

