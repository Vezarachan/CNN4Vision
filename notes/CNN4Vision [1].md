# CNN4Vision [1]

## Image Classification

### Definition

<img src="./CNN4Vision [1].assets/cat.jpg" style="zoom: 100%;" />

### The problem

Semantic gap between the semantic idea of cat and pixels - 从像素（图像）中提取其中包含的语义信息

### Challenges

1. viewpoint variation 视角变化
2. scale variation 尺度变化
3. illumination 光照
4. deformation 目标的变形
5. occlusion 遮挡
6. background clutter 背景杂物
7. Intra-class variation 类内差异

### Data-driven approach

1. Collect a dataset of images and labels - **Input**
2. Use ML to train a classifer - **training or learning a model**
3. Evaluate the classier on new images - **evaluation**

### First Classifer: Nearest Neighbor

1. Momeorize all data and labels - **train**
2. Predict the label of the most similar training image - **predict**

#### Distance Metric (used to compare image) 距离度量

##### L1 distance - depend on the coordiates of data 受数据的坐标系统的选择的影响

$$
d_1(I_1,I_2)=\sum_p|I_1^p - I_2^p|
$$

##### L2 distance

$$
d_2(I_1,I_2)=\sqrt{\sum_p(I_1^p-I_2^p)^2}
$$

#### Limitations

- kNN classifer is fast in training but flow in predicting
- distance metrics on pixels are not informative
- curse of dimensionality k近邻算法需要大量的训练数据
- Virtually, training slowly is acceptable and predicting is supposed to be fast

#### Setting Hyperparameters

交叉验证一般在较小的数据集中使用，而在深度学习中，由于计算资源的限制，不太会使用这种数据集分割方法。

<img src="./CNN4Vision [1].assets/image-20210325211238055.png" alt="image-20210325211238055" style="zoom: 33%;" />

<img src="./CNN4Vision [1].assets/image-20210325211313168.png" alt="image-20210325211313168" style="zoom:33%;" />

### Linear Classification

- basic module in neural networks
- simplest parametric approach
- can only use one template 

<img src="/Users/louxiayin/Academy/homework/CNN4Vision/notes/CNN4Vision [1].assets/image13.png" alt="查看源图像" style="zoom: 33%;" />
$$
f(x,W)=Wx + b
$$
 