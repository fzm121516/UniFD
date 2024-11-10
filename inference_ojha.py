import argparse  # 导入命令行参数解析库
import sys  # 导入系统库
import time  # 导入时间库
import os  # 导入操作系统库
import torch  # 导入PyTorch库
import torch.nn as nn  # 导入PyTorch神经网络模块
import numpy as np  # 导入NumPy库
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score  # 导入评价指标
from data import create_dataloader  # 导入数据加载函数
import random  # 导入随机数库
from models.clip_models import CLIPModel  # 导入自定义C2P_CLIP模型
from tqdm import tqdm
import argparse
from ast import arg
import os
import csv
import torch
import torchvision.transforms as transforms
import torch.utils.data
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score
from torch.utils.data import Dataset
import sys
from models import get_model
from PIL import Image 
import pickle
from tqdm import tqdm
from io import BytesIO
from copy import deepcopy
from dataset_paths import DATASET_PATHS
import random
import shutil
from scipy.ndimage.filters import gaussian_filter

def seed_torch(seed=1029):  # 设置随机种子以保证结果可重复
    random.seed(seed)  # 设置Python随机数种子
    os.environ['PYTHONHASHSEED'] = str(seed)  # 设置环境变量以影响hash函数
    np.random.seed(seed)  # 设置NumPy随机数种子
    torch.manual_seed(seed)  # 设置PyTorch随机数种子
    torch.cuda.manual_seed(seed)  # 设置GPU随机数种子
    torch.cuda.manual_seed_all(seed)  # 设置所有GPU的随机数种子（如果使用多GPU）
    torch.backends.cudnn.benchmark = False  # 禁用CUDNN的基准模式
    torch.backends.cudnn.deterministic = True  # 设置CUDNN为确定性模式
    torch.backends.cudnn.enabled = False  # 禁用CUDNN

seed_torch(123)  # 设定随机种子为123

def printSet(set_str):  # 定义打印集合信息的函数
    set_str = str(set_str)  # 将集合信息转换为字符串
    num = len(set_str)  # 计算字符串长度
    print("="*num*3)  # 打印分隔线
    print(" "*num + set_str)  # 打印集合信息
    print("="*num*3)  # 打印分隔线

def parse_args():  # 定义解析命令行参数的函数
    parser = argparse.ArgumentParser(description='test C2P-CLIP')  # 创建解析器
    parser.add_argument('--loadSize', type=int, default=256)  # 添加加载尺寸参数
    parser.add_argument('--cropSize', type=int, default=224)  # 添加裁剪尺寸参数
    # parser.add_argument('--batch_size', type=int, default=64)  # 添加批次大小参数
    parser.add_argument('--shuffle', action='store_true', default=False)
    parser.add_argument('--dataroot1', type=str, default='/home/fanzheming/zm/NPR-DeepfakeDetection/dataset/ForenSynths8test/ForenSynths')  # 添加数据根路径参数
    parser.add_argument('--dataroot2', type=str, default='/home/fanzheming/zm/NPR-DeepfakeDetection/dataset/GANGen-Detection')  # 添加数据根路径参数
    parser.add_argument('--dataroot3', type=str, default='/home/fanzheming/zm/NPR-DeepfakeDetection/dataset/DiffusionForensics8test/DiffusionForensics')  # 添加数据根路径参数
    parser.add_argument('--dataroot4', type=str, default='/home/fanzheming/zm/NPR-DeepfakeDetection/dataset/UniversalFakeDetect')  # 添加数据根路径参数
    parser.add_argument('--dataroot5', type=str, default='/home/fanzheming/zm/NPR-DeepfakeDetection/dataset/Diffusion1kStep')  # 添加数据根路径参数
    parser.add_argument('--model_name', type=str, default='ViT-L/14')  # 添加模型名称
    parser.add_argument('--model_path', type=str, default='/home/fanzheming/zm/Ojha/pretrained_weights/fc_weights.pth')  # 添加模型路径参数
    parser.add_argument('--real_path', type=str, default=None, help='dir name or a pickle')
    parser.add_argument('--fake_path', type=str, default=None, help='dir name or a pickle')
    parser.add_argument('--data_mode', type=str, default=None, help='wang2020 or ours')
    # parser.add_argument('--max_sample', type=int, default=1000, help='only check this number of images for both fake/real')

    parser.add_argument('--arch', type=str, default='CLIP:ViT-L/14')
    parser.add_argument('--ckpt', type=str, default='./pretrained_weights/fc_weights.pth')

    parser.add_argument('--result_folder', type=str, default='result', help='')
    parser.add_argument('--batch_size', type=int, default=128)

    parser.add_argument('--jpeg_quality', type=int, default=None, help="100, 90, 80, ... 30. Used to test robustness of our model. Not apply if None")
    parser.add_argument('--gaussian_sigma', type=int, default=None, help="0,1,2,3,4.     Used to test robustness of our model. Not apply if None")
    parser.add_argument('--isTrain', action='store_true')
    args = parser.parse_args()  # 解析参数

    def print_options(parser, args):  # 定义打印参数选项的函数
        message = ''
        message += '----------------- Options ---------------\n'  # 打印标题
        for k, v in sorted(vars(args).items()):  # 遍历所有参数
            comment = ''
            default = parser.get_default(k)  # 获取默认值
            if v != default:  # 如果值不等于默认值
                comment = '\t[default: %s]' % str(default)  # 添加默认值注释
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)  # 格式化输出参数
        message += '----------------- End -------------------'  # 打印结束标志
        print(message)  # 打印参数信息
    print_options(parser, args)  # 调用打印函数
    return args  # 返回解析后的参数

if __name__ == '__main__':  # 主程序入口
    opt = parse_args()  # 解析命令行参数
    DetectionTests = {  # 定义检测测试集信息
        'ForenSynths': {  # 测试集名称
            'dataroot': opt.dataroot1,  # 数据根路径
            'no_resize': False,  # 是否不调整大小
            'no_crop': False,  # 是否不裁剪
        },
        'GANGen-Detection': {  # 测试集名称
            'dataroot': opt.dataroot2,  # 数据根路径
            'no_resize': False,  # 是否不调整大小
            'no_crop': False,  # 是否不裁剪
        },        
        'DiffusionForensics': {  # 测试集名称
            'dataroot': opt.dataroot3,  # 数据根路径
            'no_resize': False,  # 是否不调整大小
            'no_crop': False,  # 是否不裁剪
        },        
        'UniversalFakeDetect': {  # 测试集名称
            'dataroot': opt.dataroot4,  # 数据根路径
            'no_resize': False,  # 是否不调整大小
            'no_crop': False,  # 是否不裁剪
        },        
        'Diffusion1kStep': {  # 测试集名称
            'dataroot': opt.dataroot5,  # 数据根路径
            'no_resize': False,  # 是否不调整大小
            'no_crop': False,  # 是否不裁剪
        },
    }

    # state_dict = torch.hub._legacy_zip_load('C2P_CLIP_release_20240901.zip', './', map_location="cpu", weights_only=False)  # 加载模型状态字典（旧方法）
    # state_dict = torch.hub.load_state_dict_from_url(opt.model_path, map_location="cpu", progress=True)  # 从URL加载模型状态字典
    model = get_model(opt.arch)
    state_dict = torch.load(opt.ckpt, map_location='cpu')
    model.fc.load_state_dict(state_dict)
    print ("Model loaded..")
    model.eval()
    model.cuda()

    # 重命名键
    new_state_dict = {}
    for k, v in state_dict.items():
        if k == "weight":
            new_state_dict["model.fc.weight"] = v
        elif k == "bias":
            new_state_dict["model.fc.bias"] = v
        else:
            new_state_dict[k] = v

    # 然后加载新的状态字典
    model.load_state_dict(new_state_dict, strict=False)

    model.cuda()  # 将模型移动到GPU
    model.eval()  # 设置模型为评估模式


    for testSet in DetectionTests.keys():  # 遍历测试集
        dataroot = DetectionTests[testSet]['dataroot']  # 获取数据根路径
        printSet(testSet)  # 打印测试集名称
        accs = []  # 初始化准确率列表
        aps = []  # 初始化平均精度列表
        r_accs = []  # 初始化真实负类准确率列表
        f_accs = []  # 初始化真实正类准确率列表
        print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))  # 打印当前时间


        # files = os.listdir(dataroot)
        # for v_id, val in enumerate(tqdm(files, desc="Evaluating files")):
        for v_id, val in enumerate(os.listdir(dataroot)):  # 遍历数据根路径下的所有文件
            opt.dataroot = '{}/{}'.format(dataroot, val)  # 更新数据根路径
            opt.classes = ''  # 设置类别（此处为空字符串）
            opt.no_resize = DetectionTests[testSet]['no_resize']  # 获取是否不调整大小的参数
            opt.no_crop = DetectionTests[testSet]['no_crop']  # 获取是否不裁剪的参数
            data_loader = create_dataloader(opt)  # 创建数据加载器
            with torch.no_grad():  # 禁用梯度计算
                y_true, y_pred = [], []  # 初始化真实标签和预测值列表
                for img, label, path in tqdm(data_loader, desc=f"Processing {val}", leave=False):
                # for img, label, path in data_loader:  # 遍历数据加载器
                    y_pred.extend(model(img.cuda()).sigmoid().flatten().tolist())  # 获取预测值并添加到列表
                    y_true.extend(label.flatten().tolist())  # 获取真实标签并添加到列表
            y_true, y_pred = np.array(y_true), np.array(y_pred)  # 转换为NumPy数组
            num_negative = np.sum(y_true == 0)
            num_positive = np.sum(y_true == 1)
            print("File: {:12} - Negative: {}, Positive: {}".format(val, num_negative, num_positive))  # 打印当前文件的真实负类和正类数量
            r_acc = accuracy_score(y_true[y_true == 0], y_pred[y_true == 0] > 0.5)  # 计算真实负类的准确率
            f_acc = accuracy_score(y_true[y_true == 1], y_pred[y_true == 1] > 0.5)  # 计算真实正类的准确率
            acc = accuracy_score(y_true, y_pred > 0.5)  # 计算总体准确率
            ap = average_precision_score(y_true, y_pred)  # 计算平均精度
            r_accs.append(r_acc)  # 添加真实负类准确率到列表
            f_accs.append(f_acc)  # 添加真实正类准确率到列表

            accs.append(acc)  # 添加准确率到列表
            aps.append(ap)  # 添加平均精度到列表
            print("({} {:12}) acc: {:.2f}; ap: {:.2f}; r_acc: {:.2f}; f_acc: {:.2f}".format(
            v_id, val, acc * 100, ap * 100, r_acc * 100, f_acc * 100))  # 打印当前文件的准确率和平均精度

        print("({} {:10}) acc: {:.2f}; ap: {:.2f}; r_acc: {:.2f}; f_acc: {:.2f}".format(
        v_id + 1, 'Mean', np.array(accs).mean() * 100, np.array(aps).mean() * 100, 
        np.array(r_accs).mean() * 100, np.array(f_accs).mean() * 100))  # 打印均值
        print('*' * 25)  # 打印分隔线
