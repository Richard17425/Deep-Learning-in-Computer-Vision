# 深度学习笔记
`Notes during my learning DL`

学习视频我看的是：[2022吴恩达机器学习Deeplearning.ai课程](https://www.bilibili.com/video/BV1Pa411X76s/?p=43&spm_id_from=pageDriver&vd_source=72cbed57f84134f653cd0ebd0e4e2cff)

***

神经网络术语：
---
- 输入层(Input layer): 输入神经网络用于训练的数据特征,也被称为第0层。
- 输出层(Output layer):输出的值即为神经网络预测的结果
- 隐藏层(Hidden layer): 中间层，大多不止一个
- 激活(Activations):负担能力意识和感知质量。
- 激活值(Activation values)


## 神经网络用于面部识别
![](images/1.png)
1. 第一个隐藏层中的神经元主要寻找垂直线边缘
2. 第二个隐藏层中的神经元则是在寻找有方向的线或者边
3. 第三个隐藏层中的神经元在这个方向上寻找一条线，依次类推...
>在较早的隐藏层中的神经元在图中寻找非常短的线条或非常短的边缘
4. 再往后的神经元将这些短线条聚集在一起，形成很多短小的线条和边缘段，从而好寻找脸的一部分
5. 神经元将人脸的不同部分聚合在一起，然后检测图像中是否存在粗粒度更大的面部轮廓
6. 最后检测这张脸与不同面部轮廓的符合程度，创建一组丰富的特征，然后帮助输出层确定图片中人物的身份
>不同层的神经元分析的实际上可以对应于图像中不同大小的区域

## 神经网络中的网络层
![](images/2.png)
![](images/3.png)

>要注意在比较复杂的神经元之间计算时角标和矢量
![](images/4.png)
![](images/5.png)

## 前向传播算法(Forward propagation algorithm)

![](images/6.png)

要传播神经元的激活值，计算是从左往右进行，

## TensorFlow
- 示例1 ![](images/7.png)

- 示例2![](images/8.png)

> - Dense是神经网络一种名字
> - unit参数是该层神经元的个数
> - 在TensorFlow中矩阵(张量tensor)的表示用 `np.array([[ ]])`两个大括号来表示
![](images/9.png)

TensorFlow中的`Sequential()`函数可以将建立的两个神经层串联在一起，用TensorFlow可以大大简化代码

![](images/10.png)  
![](images/11.png)  

对应的实验室为：[TensorFlow](Advanced_Learning_Algorithms/week1/5.TensorFlow%20implementation/C2_W1_Lab02_CoffeeRoasting_TF.ipynb)