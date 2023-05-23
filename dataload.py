"""
@Author : Chaser
@Time : 2023/3/3 14:50
@File : dataload.py
@software : PyCharm
"""
"""
@update:
2023/3/5
2023/3/10
"""

import os
import torch
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

#车的数据集加载器
class CarDataset(Dataset):
    # 传入图像地址文件夹名和标签文件夹名
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        # 包含在image_dir文件夹下的所有文件
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # 遍历读取图片的相对位置
        image_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg", "_mask.gif"))

        # 将图片转换为RGB图，将标签图片转换为灰度图
        image = np.array(Image.open(image_path).convert("RGB"),dtype=float)
        mask = np.array(Image.open(mask_path).convert("L"),dtype=float)
        #令像素为255.0的为1，使所以像素为0或1
        mask[mask == 255.0 ] = 1.0

        # 对image，mask进行空间转换等操作
        if self.transform is not None:
            augumentations = self.transform(image=image, mask=mask)
            image = augumentations["image"]
            mask = augumentations["mask"]

        # print(image,mask)
        # print(mask.max())
        return image, mask