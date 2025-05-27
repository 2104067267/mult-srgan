from utils import *
from torch.utils.data import Dataset
from torchvision import transforms
import os
import random

class TrainDatasetFromFolder(Dataset):
    def __init__(self, hr_image_dir, crop_size, upscale_factor, l):
        super(TrainDatasetFromFolder, self).__init__()
        self.hr_image_filenames = [os.path.join(hr_image_dir, x) for x in listdir(hr_image_dir)[:l] if is_image_file(x)]
        self.hr_transform = transforms.Compose([
            transforms.RandomRotation(degrees=(-5, 5)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
        ])
        self.lr_transform = transforms.Compose([
            transforms.Resize(crop_size // upscale_factor, interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ])
        self.to_pil = transforms.ToPILImage()

    def __getitem__(self, index):
        hr_image = Image.open(self.hr_image_filenames[index])
        if hr_image.mode == 'RGBA':
            hr_image = hr_image.convert('RGB')  # 将 4 通道图像转换为 3 通道图像
        hr_image = self.hr_transform(hr_image)
        radius = random.randint(2,12)
        lr_image = defocus_blur(self.to_pil(hr_image), radius)
        kernal_size = random.randint(1,5)*2+1
        sigma = random.randint(2,11)
        lr_image = transforms.GaussianBlur(kernal_size, sigma)(lr_image)
        lr_image = self.lr_transform(lr_image)
        return lr_image, hr_image

    def __len__(self):
        return len(self.hr_image_filenames)



class TestDatasetFromFolder(Dataset):
    def __init__(self, hr_image_dir,lr_image_dir,l):
        super(TestDatasetFromFolder, self).__init__()
        self.lr_image_filenames = [os.path.join(lr_image_dir, x) for x in listdir(lr_image_dir)[:l] if is_image_file(x)]
        self.to_tensor = transforms.ToTensor()

        if hr_image_dir is not None:
            self.hr_image_filenames = [os.path.join(hr_image_dir, x) for x in listdir(hr_image_dir)[:l] if is_image_file(x)]
        else:
            self.hr_image_filenames = None

    def __getitem__(self, index):
        lr_image = Image.open(self.lr_image_filenames[index])
        if lr_image.mode == 'RGBA':
            lr_image = lr_image.convert('RGB')  # 将 4 通道图像转换为 3 通道图像
        lr_image = self.to_tensor(lr_image)

        hr_image = 0
        if self.hr_image_filenames is not None:
            hr_image = Image.open(self.hr_image_filenames[index])
            if hr_image.mode == 'RGBA':
                hr_image = hr_image.convert('RGB')  # 将 4 通道图像转换为 3 通道图像
            hr_image = self.to_tensor(hr_image)

        return lr_image,hr_image



    def __len__(self):
        return len(self.lr_image_filenames)


class ValDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(ValDatasetFromFolder, self).__init__()
        self.upscale_factor = upscale_factor
        self.image_filenames = [os.path.join(dataset_dir, x) for x in listdir(dataset_dir)[:5] if is_image_file(x)]

    def __getitem__(self, index):
        hr_image = Image.open(self.image_filenames[index])
        w, h = hr_image.size
        crop_size = calculate_valid_crop_size(min(w, h), self.upscale_factor)
        lr_scale = Resize(crop_size // self.upscale_factor, interpolation=Image.BICUBIC)
        hr_scale = Resize(crop_size, interpolation=Image.BICUBIC)
        hr_image = CenterCrop(crop_size)(hr_image)
        lr_image = lr_scale(hr_image)
        hr_restore_img = hr_scale(lr_image)
        return ToTensor()(lr_image), ToTensor()(hr_restore_img), ToTensor()(hr_image)

    def __len__(self):
        return len(self.image_filenames)