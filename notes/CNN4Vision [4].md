# CNN4Vision [4]

## Convolutional Neural Networks

<img src="./CNN4Vision [4].assets/1*vlgEOOMh9UcWkRsX7OniJQ.gif" alt="See the source image"  />

### Convolution layer

<img src="./CNN4Vision [4].assets/deep-learning-into-advance-1-image-convnet-27-638.jpg" alt="See the source image"  />

$32\times 32\times 3$ images and $5\times 5\times 3$ filter (weights) $\Rightarrow\ w^Tx + b$ 

一般而言，在卷积神经网络中会采用多种卷积核，不同的卷积核可以从图像中提取出不同的模式（特征），如下图所示

<img src="./CNN4Vision [4].assets/fe7bdaec-36a4-f64e-d521-5981a52b6765.png" alt="See the source image" style="zoom:70%;" />



#### 卷积神经网络的使用

通常在卷积神经网络会采用一系列的卷积操作和激活操作的组合并组成序列，从而逐步从低层次到高层次提取出图像的特征信息。从本质上来说，这个过程的可以与传统的计算机视觉中人工构建高级特征（例如SIFT、ORB、SURF等）的过程相似，只不过这个过程是自动完成的。

<img src="./CNN4Vision [4].assets/Screenshot 2018-05-13 18.15.53.png" alt="See the source image" style="zoom: 40%;" />

<img src="./CNN4Vision [4].assets/yann-le-cun-6-638.jpg" alt="查看源图像" style="zoom: 80%;" />

进行卷积时，最终输出的图像大小可以表示为一个公式$(N-F)/S+1$，其中N为原始图像的边长，F为卷积核的边长，S为卷积的步长，不过在图像预处理中，也会进行图像边缘的填充，如零填充（zero padding）、复制填充（copy padding）等，此时公式变为$(N-F+2P)/S+1$，其中P为填充的边长长度。卷积时步长起到降采样的作用，其作用相似于池化但是比池化更加好用。

<img src="./CNN4Vision [4].assets/1*ciDgQEjViWLnCbmX-EeSrA.gif" alt="查看源图像" style="zoom:80%;" />

### Pooling Layer

- 对输入进行降采样处理
- 使得数据更小、更容易管理
- 不会改变输入数据的深度
- 不会采用填充（只是进行数据降采样）

#### Max pooling

<img src="./CNN4Vision [4].assets/OIP.gaD6SJ6kQNVOclE_WkwLNQHaDF" alt="查看源图像"  />

#### Average pooling

<img src="./CNN4Vision [4].assets/1*q0lk6B6gzvsSQSDn-20zJA.png" alt="查看源图像" style="zoom: 38%;" />

最大值池化相较于平均值池化更将常用，因为其能够最直观地表示激活的强度。

### Fully Connected Layer

<img src="./CNN4Vision [4].assets/fc.jpg" alt="fc"  />