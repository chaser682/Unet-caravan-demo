"""
@Author : Chaser
@Time : 2023/3/3 14:59
@File : debug.py
@software : PyCharm
"""

"""
@update:
2023/3/3
2023/3/5
2023/3/10
"""

# import numpy as np
# import PIL.Image as Image
#
# mask_path = "./data/train_mask/0cdf5b5d0ce1_01_mask.gif"
#
#
# mask = np.array(Image.open(mask_path).convert("L"),dtype=float)
#
# print(mask)
# import os
# folder = "./saved_images/"
# idx = "0"
#
# os.remove("%spred_%s.png" % (folder, idx))
# import torchvision
# import torch
# y = torch.randn(1,1,224,224)
# y = (y>0.5).float()*255
# x = torch.ones(1,1,224,224)
# z = torch.zeros(1,1,224,224)
# print(y)
# torchvision.utils.save_image(y, "test.png")


# import torch
# a = torch.ones(3,3,dtype=torch.int32)
# b = torch.zeros(3,3,dtype=torch.int32)
# a = a.int()
# print(a|b)
# print(a&b)


# import numpy as np
# import matplotlib.pyplot as plt
#
# a = np.random.normal(10,5,500)
# b = np.random.normal(5,5,500)
# cluster1 = np.array([[a,b,-1] for a,b in zip(a,b)])
#
# a = np.random.normal(30,5,500)
# b = np.random.normal(35,5,500)
# cluster2 = np.array([[a,b,1] for a,b in zip(a,b)])
#
# #axis 表示轴线，为1表示垂直，为0表示水平
# dataset = np.append(cluster1,cluster2,axis=0)
#
# # print(a)
# # print(b)
# # print(dataset)
#
#
# for i in dataset:
#     if i[2] == 1:
#         plt.scatter(i[0],i[1],c='r',s=8)
#     else:
#         plt.scatter(i[0],i[1],c='g',s=8)
# plt.show()


'''图像分割评价指标'''


# print('a\nb')

from train import evaluation_curve

a = [95.651878,95.933640,95.941444]
b = [90.1212,90.41232,90.141234]
c = [83.12312,82.123124,84.21312]

evaluation_curve(a,b,c)