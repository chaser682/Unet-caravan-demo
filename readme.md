## 文件目录说明
![目录结构](/declare/demo_contents.png)
  

* __data包__  
test_images为测试集input  
test_mask为测试集target  
train_images为训练集input  
train_mask为训练集target  
  

* __parameters包__  
用于保存学习率为lr训练的模型参数  
  

* __saved_images包__   
用于保存模型训练完成后的predicted图像和target图像  
  

* __dataload.py文件__    
定义了CarDataset类，Car数据集加载器  
  

* __model.py文件__  
分别定义了DoubleConv类、Down类、Up类、OutConv类和Unet类  
**DoubleConv类**：进行连续的两次卷积，每次卷积的过程为Convolution -> BN -> ReLU
**Down类**：下采样过程（downsampling），由最大池化（Maxpool）和连续两次卷积（DoubleConv）组成  
**Up类**：上采样过程（upsampling）,由反卷积（转置卷积或双线性插值）和连续两次卷积（DoubleConv）组成  
**OutConv类**：最后一次卷积输出结果  
**Unet类**：Unet网络的主体结构，具体过程是输入卷积->四次下采样->四次上采样->得到卷积输出  
  

* __utils.py文件__  
一共定义了save_checkpoint、load_checkpoint、get_loaders、check_accuracy和save_predictions5个方法  
**save_checkpoint方法**： 用于保存模型的参数  
**load_checkpoint**：用于加载模型的参数  
**get_loaders**：用于加载数据集  
**check_accuracy**：用于检查训练模型的精度  
**save_predictions**:用于保存预测影像的结果  
   

* __train.py文件__  
一共定义了train_fn、find_lr、loss_lr_curve和main四个方法  
**train_fn方法**：用于模型的训练  
**find_lr方法**：找到最佳的学习率  
**loss_lr_curve方法**：用户绘制损失函数与学习率的关联曲线  
* **loss_curve方法**：绘制损失函数与数据训练批次的关系
* **evaluation_curve方法**：绘制评估函数随着训练次数的变化  
**main方法**：主程序的入口，调用各个方法和api，完成模型的训练和预测以及评估  
  

## 数据集准备  
![数据集网站](/declare/kaggle-carvana.png)
网站链接：[Kaggle-carvana数据集下载网站](https://www.kaggle.com/competitions/carvana-image-masking-challenge/data)  
  

## 数据处理 
**先对.jpg和.tif图片转换为RGB图片何灰度图片，再使用albumentations库对数据进行增强**
### 数据加载
1.遍历读取图片的相对位置  
2.将训练数据转换为RGB图，测试数据转换为灰度图  
3.对数据进行增强操作  

### 数据增强  
导入albumentations数据增强库，该库是负责处理图像的一个库，可用于所有数据类型，支持各类  
图像处理的方法，图像翻转、裁剪、填充、组合序列化等等操作，与此同时，该库也是处理图像最快的库
* **Compose方法：** 将所有图形转换操作组合起来
* **Resize方法：** 重置图形的尺寸
* **Rotate方法：** 对图形进行旋转
* **HorizontalFlip方法：** 图形围绕Y轴水平翻转
* **VerticalFlip方法：** 图形围绕X轴垂直翻转
* **Normalize方法：** 图形进行归一化处理
* **ToTensorV2方法：** 将图形数据转换为tensor数据


## 模型构建  
该模型使用的是语义分割的经典网络Unet网络。UNet是一种用于图像分割的卷积神经网络，它结合了编码器和解码器，
可以有效地将输入图像分割成多个部分。UNet最初是用于生物医学图像分割，但现在已经广泛应用于其他领域，
如自然图像分割、语义分割等。  
UNet的编码器部分由卷积层和池化层组成，用于提取图像的特征。解码器部分由反卷积层和跳跃连接组成，
用于将编码器提取的特征映射还原为原始图像大小。跳跃连接是指将编码器的特征图与解码器的特征图连接起来，
以便解码器可以使用更多的上下文信息进行分割。   

**Unet网络**
![Unet网络结构](/declare/unet.png)  
  
该网络结构分为Encoder和Decoder两部分：
### Encoder
Encoder由卷积操作和下采样操作组成，每次卷积的卷积结构为 3x3 的卷积核，padding 为 0 ，striding 为 1；
下采样操作为最大池化（max pooling），stride为2，输出大小为1/2 *(H, W)。下采样操作进行四次，同时每层进行
两次卷积操作，将得到的特征图输入Decoder。  

### Decoder
Decoder由卷积操作和上采样操作组成，上采样的方式为双线性插值方式实现；上采样操作完成后，将得到的特征图与Encoder同层的
特征图进行跳跃连接，即进行拼接，再进行两次卷积操作。重复四次该过程，最后再进行一次卷积，得到最终的特征图。

## 模型训练
1.定义超参数（lr、device、batch_size、epochs等等）  
2.实例化模型、实例化损失函数和优化器  
3.训练集和数据集的加载  
4.是否使用预训练参数  
5.模型训练  
6.模型评估  
7.参数保存  
8.可视化结果  


## 模型评估  
* **像素准确率：** 在图像分割中，分类正确的像素数量占总像素数量的比例  
计算公式：**像素准确率 = 分类正确的像素数 / 总像素数**
* **Dice分数：** 图像分割评估指标，用于评估分割结果的准确性，它计算了预测分割结果和真实分割结果之间的相似度  
计算公式：**DICE = \frac{2TP}{2TP+FP+FN}** 
* **IOU分数：** 一种用于评估图像分割结果的指标。它是通过计算分割结果与真实标注之间的交集与并集之比来衡量分割的准确性  
计算公式：**IOU = (分割结果与真实标注的交集面积) / (分割结果与真实标注的并集面积)**


## 项目运行
python train.py


## update
如果对该项目有疑问，可以创建issue


