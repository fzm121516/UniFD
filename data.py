import torch  # 导入 PyTorch 库
import numpy as np  # 导入 NumPy 库
import torchvision.datasets as datasets  # 导入 torchvision 数据集模块
from tqdm import tqdm  # 导入进度条显示模块
from typing import Any, Callable, cast, Dict, List, Optional, Tuple  # 导入类型提示相关模块
from PIL import ImageFile, Image  # 导入 PIL 库的图像处理模块
import torchvision.transforms as transforms  # 导入 torchvision 图像变换模块
import os  # 导入操作系统相关模块

# 允许加载截断的图像文件
ImageFile.LOAD_TRUNCATED_IMAGES = True
# 定义支持的图像文件扩展名
IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")

def pil_loader(path: str) -> Image.Image:
    # 打开路径作为文件以避免 ResourceWarning
    with open(path, "rb") as f:
        img = Image.open(f)  # 打开图像文件
        return img.convert("RGB")  # 转换为 RGB 格式并返回

class ImageFolder2(datasets.DatasetFolder):
    # 自定义数据集类，继承自 DatasetFolder
    def __init__(self, root: str, transform: Optional[Callable] = None):
        # 初始化函数，设置根目录和变换
        super().__init__(
            root,
            transform=transform,
            extensions=IMG_EXTENSIONS,  # 支持的文件扩展名
            loader=pil_loader  # 使用自定义图像加载函数
        )

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        # 重写 __getitem__ 方法以返回样本、标签和路径
        path, target = self.samples[index]  # 获取样本的路径和标签
        sample = self.loader(path)  # 使用加载器加载样本
        if self.transform is not None:
            sample = self.transform(sample)  # 应用变换（如果有）
        if self.target_transform is not None:
            target = self.target_transform(target)  # 应用目标变换（如果有）
        return sample, target, path  # 返回样本、标签和路径

def get_dataset(opt):
    dset_lst = []  # 存储找到的数据集列表
    depth = 0  # 深度计数

    def get_01(root_path, depth):
        # 递归查找数据集的内部函数
        if depth > 10:
            print('=' * 10, f'\ndataset search depth max 10\n', '=' * 10)
            return  # 最大深度限制
        classes = os.listdir(root_path)  # 获取当前目录下的所有类
        if '0_real' in classes and '1_fake' in classes and len(classes) == 2:
            # 如果找到包含 0_real 和 1_fake 的文件夹
            dset_lst.append(dataset_folder(opt, root_path))  # 将数据集添加到列表
            return
        else:
            depth += 1  # 深度增加
            for cls in classes:
                get_01(root_path + '/' + cls, depth=depth)  # 递归查找子文件夹

    get_01(opt.dataroot, depth=depth)  # 从根目录开始查找数据集
    assert len(dset_lst) > 0  # 确保找到至少一个数据集
    if len(dset_lst) == 1:
        return dset_lst[0]  # 如果只有一个数据集，直接返回
    else:
        return torch.utils.data.ConcatDataset(dset_lst)  # 否则合并多个数据集并返回

import math
def translate_duplicate(img, cropSize):
    if min(img.size) < cropSize:
        width, height = img.size
        
        new_width = width * math.ceil(cropSize/width)
        new_height = height * math.ceil(cropSize/height)
        
        new_img = Image.new('RGB', (new_width, new_height))
        for i in range(0, new_width, width):
            for j in range(0, new_height, height):
                new_img.paste(img, (i, j))
        return new_img
    else:
        return img
    
def dataset_folder(opt, root_path):
    # 创建数据集的变换流程
    if opt.isTrain:
        crop_func = transforms.RandomCrop(opt.cropSize)
    elif opt.no_crop:
        crop_func = transforms.Lambda(lambda img: img)
    else:
        crop_func = transforms.CenterCrop(opt.cropSize)

    # if opt.isTrain and not opt.no_flip:
    #     flip_func = transforms.RandomHorizontalFlip()
    # else:
    #     flip_func = transforms.Lambda(lambda img: img)
    
    if opt.no_resize:
        # rz_func = transforms.Lambda(lambda img: img)
        rz_func = transforms.Lambda(lambda img: translate_duplicate(img, opt.cropSize))
    else:
        # rz_func = transforms.Lambda(lambda img: custom_resize(img, opt))
        # rz_func = transforms.Resize((opt.loadSize, opt.loadSize))
        rz_func = transforms.Resize(opt.loadSize)

    # dset = datasets.ImageFolder(
    #         root,
    #         transforms.Compose([
    #             rz_func,
    #             # transforms.Lambda(lambda img: data_augment(img, opt)),
    #             crop_func,
    #             flip_func,
    #             transforms.ToTensor(),
    #             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #         ]))

    return ImageFolder2(
        root=root_path,
        transform=transforms.Compose([
            # transforms.Resize(opt.loadSize),  # 调整图像大小
            rz_func,
            # transforms.CenterCrop(opt.cropSize),  # 中心裁剪
            crop_func,
            transforms.ToTensor(),  # 转换为张量
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),  # 标准化
        ])
    )

def create_dataloader(opt):
    dataset = get_dataset(opt)  # 获取数据集

    # 创建数据加载器
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,  # 批量大小
        shuffle=opt.shuffle,  # 不打乱数据
        sampler=None,  # 不使用额外的采样器
        num_workers=8  # 使用 8 个工作线程
    )
    return data_loader  # 返回数据加载器


