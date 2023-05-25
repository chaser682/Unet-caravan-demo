"""
@Author : Chaser
@Time : 2023/3/3 14:51
@File : train.py
@software : PyCharm
"""
"""
@update:
2023/3/5
2023/3/10
2023/4/21
2023/5/10
2023/5/20
"""
import math

import torch
import albumentations as A  # 数据增强库，对目标进行多种空间转换
from albumentations.pytorch import ToTensorV2  # 只会[h, w, c] -> [c, h, w]，不会将数据归一化到[0, 1]
from tqdm import tqdm  # python进度条库
import torch.nn as nn
import torch.optim as optim  # 导入优化器
import matplotlib.pyplot as plt
import import_ipynb
from model import Unet
from utils import (load_checkpoint,
                   save_checkpoint,
                   get_loaders,
                   check_accuracy,
                   save_predictions)
# 防止内核挂掉
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 训练的参数设置
"""
根据学习率lr与损失值loss之间的关系曲线
lr取值适合1e-3,1e-4,1e-5,1e-6
"""
lr = 0.000001  # 学习率
device = "cuda" if torch.cuda.is_available() else "cpu"  # 在gpu上训练
batch_size = 4  # 批处理大小
epochs = 3  # 训练的次数
num_workers = 1  # 工作线程数
# 图片尺寸
image_height = 224
image_width = 224
pin_menory = True
load_model = True
train_img_dir = "./data/train_images/"
train_mask_dir = "./data/train_mask/"
test_img_dir = "./data/test_images/"
test_mask_idr = "./data/test_mask/"


# 训练函数
def train_fn(loader, model, optimizer, loss_fn, scaler, losses):  # 数据读取，网络模型，优化器，损失函数，缩放标量,损失值列表
    # 读取数据
    loop = tqdm(loader)

    for idx, (data, targets) in enumerate(loop):
        # 将数据展开到GPU中
        data = data.to(device)
        targets = targets.float().unsqueeze(1).to(device)

        # optimizer.zero_grad()  # 梯度清零
        # # forward
        # with torch.cuda.amp.autocast():  # 混合精度加速训练
        #     preds = model(data)
        #     # print(data.size())
        #     # print(preds)
        #     loss = loss_fn(preds, targets)  # 计算损失
        #     # print(loss)
        # # backward
        # scaler.scale(loss).backward()  # 反向传播
        # scaler.step(optimizer)  # 优化器参数更新
        # scaler.update()  # 更新缩放标量以使其适应训练的梯度

        preds = model(data)
        loss = loss_fn(preds, targets)
        # print(preds)
        # print(targets)
        # print(targets.max())
        # 梯度清零
        optimizer.zero_grad()
        # 反向传播，梯度更新
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        loop.set_postfix(loss=loss.item())  # 设置进度条右边显示的信息
        # loss值为nan，则说明梯度爆炸或者学习率过高
        # print(data)
        # print(targets)

        # break#跑一批次数据，debug


# 找到最佳的学习率
def find_lr(loader, model, optimizer, loss_fn, device=device, init_value=1e-8, final_value=10., beta=0.98):
    num = len(loader) - 1
    mult = (final_value / init_value) ** (1 / num)
    lr = init_value
    # 动态调整学习率
    # 长度为6的字典，分别为['amsgrad', 'params', 'lr', 'betas', 'weight_decay', 'eps']
    optimizer.param_groups[0]['lr'] = lr
    avg_loss = 0
    best_loss = 0
    batch_num = 0
    losses = []
    log_lrs = []
    for x, y in loader:
        batch_num += 1
        x = x.to(device)
        # 给y在dim=1加上一维
        y = y.to(device).unsqueeze(1)
        optimizer.zero_grad()
        preds = model(x)
        loss = loss_fn(preds, y)
        # 得到平滑的损失函数值
        avg_loss = beta * avg_loss + (1 - beta) * loss.item()
        smooth_loss = avg_loss / (1 - beta ** batch_num)
        # 如果损失值爆炸，则结束
        if batch_num > 1 and smooth_loss > 4 * best_loss:
            return log_lrs, losses
        if smooth_loss < best_loss or batch_num == 1:
            best_loss = smooth_loss
        # 将损失值和学习率指数保存
        losses.append(smooth_loss)
        log_lrs.append(math.log10(lr))
        # 梯度下降
        loss.backward()
        optimizer.step()
        lr *= mult
        optimizer.param_groups[0]['lr'] = lr
    return log_lrs, losses


# 损失函数与学习率的关联曲线
def loss_lr_curve(loader, model, optimizer, loss_fn):
    logs, losses = find_lr(loader=loader, model=model, optimizer=optimizer, loss_fn=loss_fn)
    plt.xlabel("lr")
    plt.ylabel("loss")
    plt.plot(logs[10:], losses[10:])
    plt.show()

#损失函数随着训练批次的变化
def loss_curve(losses):
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.plot(losses)
    plt.show()

#准确率、dice分数、iou分数随着训练次数的变化
def evaluation_curve(accuracy,dice,iou):

    fig, ax = plt.subplots()  # 创建图实例
    ax.plot(accuracy, label='accuracy')
    ax.plot(dice, label='dice')
    ax.plot(iou, label='iou')
    ax.set_xlabel('epochs')
    ax.set_ylabel('score')
    ax.set_title("evaluation curve")
    ax.legend()
    plt.show()


# 主程序入口
def main():
    # 对训练的数据进行处理，数据增强处理
    train_transform = A.Compose(
        [
            A.Resize(height=image_height, width=image_width),  # 重置图片尺寸
            A.Rotate(limit=35, p=1.0),  # 旋转，limit表示旋转范围，p表示概率
            A.HorizontalFlip(p=0.5),  # 围绕Y轴水平翻转
            A.VerticalFlip(p=0.5),  # 围绕X轴垂直翻转
            A.Normalize(
                mean=[0.0, 0.0, 0.0],  # 归一化处理
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0),
            ToTensorV2(),
        ]
    )

    # 对测试的数据进行处理，数据增强处理
    test_transform = A.Compose(
        [
            A.Resize(height=image_height, width=image_width),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ]
    )

    # 定义模型的一些参数
    model = Unet(in_channels=3, out_channels=1).to(device)  # 实例化模型
    # 相比于BCE()，会进行sigmoid操作
    loss_fn = nn.BCEWithLogitsLoss()  # 二分类交叉熵损失函数实例化
    optimizer = optim.Adam(model.parameters(), lr=lr)  # 自适应动量算法优化器

    # 训练集和测试集的加载
    train_loader, test_loader = get_loaders(
        train_dir=train_img_dir,
        train_maskdir=train_mask_dir,
        test_dir=test_img_dir,
        test_maskdir=test_mask_idr,
        batch_size=batch_size,
        train_transform=train_transform,
        test_transform=test_transform,
        num_workers=num_workers,
        pin_memory=pin_menory
    )

    '''由于开始没有训练好的参数pth文件,故需要注释掉'''
    # 使用之前训练的参数
    if load_model:
        load_checkpoint(torch.load("./parameters/%fmy_checkpoint.pth.tar" % (lr,)), model)

    scaler = torch.cuda.amp.GradScaler()  # torch.cuda.amp提供了可以使用混合精度的方便方法，以加速训练

    #损失值列表
    losses = []
    #准确率、dice分数、iou分数
    accuracy = []
    dice = []
    iou = []

    # 训练模型，训练次数为epochs
    for epoch in range(epochs):
        train_fn(train_loader, model, optimizer, loss_fn, scaler, losses)

        # 保存模型、优化器参数
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(state=checkpoint, filename="./parameters/%fmy_checkpoint.pth.tar" % (lr,))

        # 检验模型的精度
        x,y,z = check_accuracy(test_loader, model, device)

        accuracy.append(x.cpu())
        dice.append(y.cpu())
        iou.append(z.cpu())

        # 可持久化存储，保存预测影像
        save_predictions(test_loader, model, folder="./saved_images/", device=device)

    # 画出学习率与损失函数之间的曲线，以便发现最佳学习率
    # loss_lr_curve(loader=train_loader,model=model,optimizer=optimizer,loss_fn=loss_fn)

    # 画出损失函数与训练次数之间的曲线
    loss_curve(losses)

    #绘制评估曲线
    evaluation_curve(accuracy,dice,iou)



if __name__ == "__main__":
    main()
