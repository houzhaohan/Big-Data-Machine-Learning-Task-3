# Big-Data-Machine-Learning-Task-3
### Classification of 17 types of flowers based on VGG  基于VGG的17类花卉分类
大数据及农业应用-课程作业\
大三上

### 一、实验目的
基于VGG的17类花卉分类\
花卉分类作为植物学领域中最常见的基础工作，是对花卉种类习性等深入研究的必要条件。目前已知有超过250000种开花植物，大约可分为350个科。花卉种类繁多，且这其中很多花卉具有相同特征，不同类型的花卉可能具有相似的颜色、形状和外观等特征。对于没有深入了解植物花卉的普通人来说，脱离植物学专业人士引导后能够独立识别常见的花卉仍是一个具有较大难度的工作。\
基于以上背景利用卷积神经网络帮助进行花卉分类。

### 二、实验原理
（1）图像分类方法\
传统机器学习算法：\
• 问题迁移，即将多标签分类问题转化为单标签分类问题，如将标签转化为向量、训练多个分类器等；\
• 根据多标签特点，提出新的适应性算法，包括KNN、SVM、Decision Tree等。\
深度学习算法：CNN(1980s)，AlexNet(2012)，VGG(2014)等多种图像分类算法。\
CNN-Convolutional Neural Network\
VGGNet是牛津大学计算机视觉组（Visual Geometry Group）和谷歌 DeepMind一起研究出来的深度卷积神经网络，因而冠名为VGG。\
（2）数据集介绍\
数据集为17 Category Flower Dataset，是牛津大学Visual Geometry Group选取的在英国比较常见的17种花；其中每种花有80张图片，整个数据集有1360张图片。\
（3）算法介绍\
本次所用的vgg16模型是卷积神经网络的一种，包括13个卷积层和3个全连接层。卷积过程是使用1个卷积核，在每层像素矩阵上不断按步长扫描下去，每次扫到的数值会和卷积核中对应位置的数进行相乘，然后相加求和，得到的值将会生成一个新的矩阵。 \
卷积核相当于卷积操作中的一个过滤器，用于提取我们图像的特征，卷积核的大小一般选择3x3和5x5，比较常用的是3x3，训练效果会更好。\
（4）激活函数\
ReLU函数：\
• 解决了梯度消失的问题 (在正区间)\
• 计算速度非常快，只需要判断输入是否大于0\
• 收敛速度远快于sigmoid和tanh\
• 缓解了过拟合问题的发生\
Softmax函数：\
$\mathrm{Softmax}(z_i)=\frac{e^{z_i}}{\sum_{c=1}^Ce^{z_c}}$\
引入指数形式的优点，使用指数形式的Softmax函数能够将差距大的数值拉的更大，适用于多分类问题。

### 三、实验步骤
（1）图片读取\
• 建立空的列表X,Y\
• 遍历文件夹，用resize()函数将图片大小缩放到同一尺寸\
• 用append（）函数将图像数据添加到列表X中。\
• 用append（）函数将对应的文件名称添加到列表Y中。\
（2）数据处理\
数据处理代码如下图所示，\
• 用np.array()函数将图片信息转换为矩阵形式\
• 用np.save()函数将数据存储为npy文件\
• 用np.load()函数再次读取文件\
• 建立字典，将花卉文件名进行类别分类\
（3）数据集划分\
根据需求对数据集进行划分，这里划定训练集占80%，测试集占20%，同时希望每次划分的数据集结果都是一样的，将random_state参数设置为1。\
(rs保持初始值和最后一个值，并随机化其余值)\
(rs = 0只是对所有数据执行正常随机)\
（4）模型建立\
• 将层的列表传递给Sequential的构造函数，来创建一个Sequential模型\
• 利用model.add()函数添加层。\
• 利用Model.compile()函数完成模型训练的BP模式设置\
• 注意：最后一层根据自己需要分类选择dense的节点数量，这里设置为17\
（5）模型训练\
• 为了防止过拟合加入EarlyStopping()函数，在训练过程中，评价指标不在上升，将提前结束训练。\
• 利用ModelCheckpoint()实现断点续训功能。\
• 利用model.fit()函数进行模型训练。\
• 将最终的模型保存为”model1.h5”,并用于之后的分类。\
（6）模型预测\
• 利用predict()函数进行模型预测\
• 调用confusion_matrix()函数调用混淆矩阵，作为模型的评价指标\
（7）预测界面\
• 运行上述所示模型VGG，完成图像的训练。\
• 建立新的main文件，编写预测代码。\
• 建立新的UI文件进行预测界面设置，采用PyQt5设计，完成图片加载和识别功能。

### 四、实验结果
![image](https://github.com/user-attachments/assets/84f49099-1ebb-48dc-90c6-a88b89e880e2)\
训练model1.h5模型，并用于之后的分类

![image](https://github.com/user-attachments/assets/85817d93-62e9-4109-a0d6-369c2ffe8240)\
训练VGG模型得到方程参数混淆矩阵

![image](https://github.com/user-attachments/assets/23101183-1028-42fc-aaa2-77cbe9ecd799)\
直接调用主函数识别flowers/image_0073.jpg图片为第16类

![image](https://github.com/user-attachments/assets/52e68a47-372e-40a7-998f-a97e70532ae2)\
使用人机交互程序UI.py进行预测界面设置（采用PyQt5设计）加载图片开始识别，得到flowers/image_0073.jpg图片为第16类。

### 五、实验总结
实验结果显示，VGG16模型在17类花卉分类任务中表现出色，达到了较高的分类准确性。这是基于卷积神经网络强大的特征提取能力和VGG16模型较深的网络结构。\
通过将数据集划分为训练集和测试集，实验验证了模型的泛化能力。结果显示，模型在测试集上的表现与训练集相近，说明模型具有较好的泛化能力。\
实验结果表明，卷积神经网络在花卉分类任务中具有强大的特征提取能力和分类准确性。同时，通过不断优化模型训练过程，可以进一步提升模型的性能。
