"""
@Author : Chaser
@Time : 2023/3/3 14:53
@File : utils.py
@software : PyCharm
"""
"""
@update:
2023/3/5
2023/3/10
2023/5/20
"""

import import_ipynb
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from dataload import CarDataset
from torch.utils.data import DataLoader
import os


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# print(device)

# 保存训练模型参数
def save_checkpoint(state, filename):
    print("-> Saving checkpoint")
    torch.save(state, filename)


# 加载模型参数
def load_checkpoint(checkpoint, model):
    print("-> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


# 加载数据集
def get_loaders(
        train_dir,
        train_maskdir,
        test_dir,
        test_maskdir,
        batch_size,  # 每个批次的大小
        train_transform,
        test_transform,
        num_workers=4,  # 线程数
        pin_memory=True  # 拷贝数据到 CUDA Pinned Memory
):
    # 训练集
    train_dataset = CarDataset(image_dir=train_dir,
                               mask_dir=train_maskdir,
                               transform=train_transform)
    # 读取训练数据
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              pin_memory=pin_memory,
                              shuffle=True)  # 打乱数据集顺序

    # 测试集
    test_dataset = CarDataset(image_dir=test_dir,
                              mask_dir=test_maskdir,
                              transform=test_transform)
    # 读取测试数据
    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             pin_memory=pin_memory,
                             shuffle=True)

    # idata = iter(train_loader)
    # print(next(idata))
    return train_loader, test_loader


# 检测训练模型的精度
def check_accuracy(loader, model, device="cuda"):
    # 初始化正确率
    num_correct = 0
    # 初始化总像素
    num_pixels = 0
    # 每次训练的得分
    dice_score = 0
    # 位置偏差得分
    iou_score = 0

    # 不启用 Batch Normalization 和 Dropout（预测之前一定要进行这一步）
    model.eval()

    # 评估模型时不需要记录梯度数据
    with torch.no_grad():
        for x, y in loader:
            # 将tensor转换为cuda张量
            x = x.to(device)
            y = y.to(device).unsqueeze(1)  # 在维度为1的位置插入一个维度
            preds = torch.sigmoid(model(x))
            # print(preds.max())
            preds = (preds > 0.5).float()  # 将大于0.5的设置为1,否则为0

            num_correct += (preds == y).sum()  # 统计相同结果的像素量
            num_pixels += torch.numel(preds)  # 总像素

            # 防止0除
            smooth = 1e-8
            # Dice相似系数计算公式:dice = (2*tp)/(fp+2*tp+fn)
            dice_score += (2.0 * (preds * y).sum() + smooth) / ((preds + y).sum() + smooth)
            # print(dice_score)

            # iou计算公式：iou = tp/(tp+fp+fn)
            preds = preds.int()
            y = y.int()
            iou_score += (1.0 * (preds & y).sum() + smooth) / ((preds | y).sum() + smooth)

    print("The accuracy of the Unet is %.6f" % (num_correct / num_pixels * 100))
    print("Dice score : %.6f" % (dice_score / len(loader)))
    print("IOU score : %.6f" % (iou_score / len(loader)))

    # 启用 Batch Normalization 和 Dropout
    model.train()


# 保存预测影像结果
def save_predictions(loader, model, folder="./saved_images/", device="cuda"):
    model.eval()

    for idx, (x, y) in enumerate(loader):
        x = x.to(device)
        y = y.unsqueeze(1).float().to(device)
        with torch.no_grad():
            x = model(x)
            preds = torch.sigmoid(x)
            preds = (preds > 0.5).float()
        # # 保存结果影像
        # if os.path.exists("%spred_%s.png" % (folder, idx)):
        #     os.remove("%spred_%s.png" % (folder, idx))
        # torchvision.utils.save_image(preds, "%spred_%s.png" % (folder, idx))
        # # 标签影像
        # if os.path.exists("%s%s.png" % (folder, idx)):
        #     os.remove("%s%s.png" % (folder, idx))
        # # print(x.size(),x.max())
        # # print(x)
        # torchvision.utils.save_image(y, "%s%s.png" % (folder, idx))

        # 保存结果影像和标签影像
        # print(preds.size(),y.size())
        img = torch.cat([preds, y], 0)
        if os.path.exists(os.path.join(folder, f"img_{idx}.png")):
            os.remove(os.path.join(folder, f"img_{idx}.png"))
        torchvision.utils.save_image(img.cpu(), os.path.join(folder, f"img_{idx}.png"))

        # 查看标签图像
        # for i in range(x.size(0)):
        #     transforms.ToPILImage()(x[i]).save(os.path.join(folder, f"img_input_{idx}_{i}.png"))
        # break
