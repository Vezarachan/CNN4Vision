# CNN4Vision [5]

## How to Train Neural Networks

### Activation Function

#### Sigmoid

$$
\sigma(x)=\frac{1}{1+e^{-x}}\\
\sigma(x)\in [0,1]\\
\frac{d\sigma (x)}{dx}=\sigma(x)\left(1-\sigma(x)\right)
$$

<img src="/Users/louxiayin/Academy/homework/CNN4Vision/notes/CNN4Vision [5].assets/1*6A3A_rt4YmumHusvTvVTxw.png" alt="查看源图像" style="zoom: 23%;" />

##### Problems:

1. Saturated neurons "kill" the gradient - 梯度消失问题（当输出过大或过小时，其梯度趋向于0）
2. Sigmoid output is not zero-centered - Sigmoid的梯度总是大于0，从而无法通过正负来逼近最佳的路径，这使得参数更新速度变慢
3. exp() is a bit compute expensive - 指数运算耗费计算资源
4. 一般在神经网络中不会使用sigmoid，一个是该激活函数固有的饱和问题，另一个则是计算复杂性的问题

#### tanh

$$
\tanh(x)=\frac{\sinh (x)}{\cosh(x)}=\frac{e^x - e^{-x}}{e^x + e^{-x}}
$$

<img src="./CNN4Vision [5].assets/tanh.jpg" alt="查看源图像" style="zoom:50%;" />

##### Problems:

1. tanh kills gradients when saturated - 同样存在梯度消失的现象
2. $\tanh$是一个零中心函数

#### ReLU

$$
f(x)=\max(0,x)
$$



<img src="./CNN4Vision [5].assets/0*n_ZGycAljU90iweS.png" alt="查看源图像" style="zoom:40%;" />



1. 非常简单，具有很高的计算效率
2. 具有相较于sigmoid和tanh更高的收敛速度
3. 不会产生饱和的现象
4. 并非是0中心的（zero-centered）
5. 仍旧会出现梯度消失的现象（左半边的梯度为0）

#### Leaky ReLU & Parametric Rectifier (PReLU)

$$
f(x)=\max(0.01x, x)\\
f(x)=\max(\alpha x, x)
$$

<img src="./CNN4Vision [5].assets/1*ypsvQH7kvtI2BhzR2eT_Sw.png" alt="查看源图像" style="zoom:40%;" />

1. 不会产生饱和现象
2. 较高的计算效率
3. 更快的收敛速度

#### Exponential Linear Units (ELU)

$$
f(x)=\left\{
\begin{matrix}
x & if\ x> 0\\
\alpha(\exp(x)-1) & if\ x\le 0
\end{matrix}
\right.
$$

<img src="./CNN4Vision [5].assets/ELU.png" alt="查看源图像" style="zoom:25%;" />

#### Maxout

$$
\max(w_1^Tx + b_1, w_2^Tx + b_2)
$$

1. 不会饱和
2. 不会梯度消失
3. 但是参数数量会翻倍
4. ReLU和Leaky ReLU的泛化版本

### Data Preprocessing

Normalization - 归一化处理

PCA、Whitening 处理 -> 降维处理（图像一般需要保留原有结构，因此并不会采用）

一般在图像中，比较多的预处理只是进行零中心化，即减去图像数据的均值

### Weight Initialization

`权重初始化的关键在于破除神经元对称性，增强其多样性，避免梯度过低或者产生相同的梯度。`

#### all zero initialization

```python
w = np.zeros(D)
```

假如将所有的权重都置为0

1. 那么则可能会导致神经元消失
2. 导致很多神经元得到相同的梯度，降低了神经元的多样化

#### small random numbers

works  for small networks but can cause problems in deeper networks

随着每一层的权重相乘，其分布的会逐渐缩小，到后面的层收敛至0

```python
w = o.o1 * np.random.randn(D, H)
```

#### calibrating the variances with 1/sqrt(n)

```python
w = np.random.randn(n) / sqrt(n)
```

#### sparse initialization 稀疏初始化

将权重矩阵的值置为0，并随机选择神经元进行连接（通常要连接的神经元可能只有10个）。

#### sparse initialization

稀疏初始化将权重矩阵的值置为0，然后随机选择少数的神经元进行连接。

#### [MSRA/He initialization](https://openaccess.thecvf.com/content_iccv_2015/papers/He_Delving_Deep_into_ICCV_2015_paper.pdf)

`对于使用ReLU激活函数的神经网络，一般采用如下的参数初始化：`

```python
w = np.random.randn(n) * sqrt(2.0 / n)
```

### Regularization

` 正则化的主要目的是抑制模型的过拟合效应，其通过对参数的复杂度进行惩罚来实现。`

L2正则化项在模型中一般表示为$\lambda w^2$，但是为了计算更加方便，一般会乘以0.5，如下式所示
$$
J(w)=\frac{1}{2}\lambda w^2\\
J‘(w)=\lambda w
$$
L1正则化一般表示为$\lambda |w|$，此外，模型中经常为结合L1和L2正则化（又称弹性网络正则化，elastic net regularization），如下式所示
$$
J(w)=\lambda_1 |w| + \lambda_2 w^2
$$
最大规范约束（max norm constraints）也是一种正则化的方法，其主要是确保每个神经元的模始终处在一个范围内，即
$$
\|w\| < c
$$
这种正则化方法的一个显著优点是即使在较大的学习率的情况下，神经网络依然不会“爆炸”（因为参数更新始终是有一定界限的）。

dropout也是一种非常有效的、简单的正则化手段，在dropout层中，神经元之间的连接会以一定的概率p被抛弃，从而增加了随机性，增强了整个神经网络的泛化性能。

<img src="./CNN4Vision [5].assets/dropout.png" alt="查看源图像" style="zoom: 33%;" />



### Talk about Loss Function Again

#### Classification

##### SVM

在一些论文中提及使用平方合页损失（squared hinge loss）相较于普通的合页损失具有更佳的表现
$$
L_i=\sum_{j\neq y_i}\max(0, f_j - f_{y_i} + 1)\\
L_i=\sum_{j\neq y_i}\max(0, (f_j - f_{y_i} + 1)^2)
$$

##### Softmax

$$
L_i = -\log(\frac{e^{f_{y_i}}}{\sum_je^{f_j}})
$$

softmax存在一个问题，即指数运算在较多类别时会比较消耗计算资源，因此在一些特定应用（比如自然语言处理）中会采用hierarchical softmax。

#### Attribute classification

attribute classification表示对于输入$x_i$而言，并不存在唯一的答案$y_i$，而是具有多个属性。其中一种损失函数如下式所示，其会惩罚分类错误的点或者分数小于1。
$$
L_i = \sum_j \max (0, 1-y_{ij}f_j)
$$
另一种损失函数是为每个属性（attribute）训练一个逻辑斯谛回归分类器，其有两类0或1，当$P$大于0.5时属于1，而小于0.5时则属于0
$$
P(y=1|x;w,b)=\frac{1}{1+e^{-(w^Tx + b)}}=\sigma(w^Tx + b)\\
P(y=0|x;w,b)=\frac{e^{-(w^Tx + b)}}{1+e^{-(w^Tx + b)}}
$$
故其损失函数为
$$
L_i = -\sum_j y_{ij}\log(\sigma(f_j)) + (1-y_{ij})\log (1-\sigma(f_j))
$$

#### Regression

##### L2 norm loss

$$
L_i=\|f - y_i\|_2^2
$$

##### L1 norm loss

$$
L_i = \|f-y_i\|_1 = \sum_j |f_j-(y_i)_j|
$$

L2损失函数存在较大的问题，首先是其较难优化，其次是其非常容易受到异常值（离群点）的影响。一种解决方法是将不同的值量化为多个类别，然后按分类进行处理。

### Batch Normalization

1. 进行归一化
2. 如有需要，则通过学习缩放和平移参数还原

$$
\hat{x}^{(k)}=\frac{x^{(k)}-E[x^{(k)}]}{\sqrt{Var[x^{(k)}]}}\\
y_i = \gamma \hat{x}+\beta
$$

注：$\gamma$和$\beta$是需要学习的参数

**批量归一化层通常位于全连接层或者卷积层之后，或者非线性化层之前**

#### 优点

1. 改善通过网络的梯度流
2. 允许更高的学习率
3. 减少对于初始化的强依赖性
4. 可以看作是一种正则化
5. 更多的用在标准卷积神经网络

<img src="./CNN4Vision [5].assets/batch_normalization.png" alt="查看源图像"  />

### Layer Normalization

$$
\mu^l = \frac{1}{H}\sum_{i=1}^H a_i^l\\
\sigma^l = \sqrt{\frac{1}{H}\sum_{i=1}^H (a_i^l - \mu^l)^2}
$$

其中H为层中的隐藏单元的数量

<img src="./CNN4Vision [5].assets/groupnorm-gn-zu.png" alt="查看源图像"  />

### Hyperparameter Optimization

#### Cross- validation strategy 交叉验证

首先通过几次epoch粗略的发现哪些参数对于准确度具有影响，而后再通过训练更加精细地寻找最佳参数

#### Searching for better parameters

在log空间中搜寻最佳参数非常有效。相对于采用固定间隔的参数搜索，随机参数搜索更容易找到更好的参数设定（如下图所示）。

<img src="./CNN4Vision [5].assets/Comparison-between-a-grid-search-and-b-random-search-for-hyper-parameter-tuning-The.png" alt="查看源图像" style="zoom:67%;" />

 

