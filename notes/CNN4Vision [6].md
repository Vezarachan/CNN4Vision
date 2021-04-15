# CNN4Vision [6]

## How to Train a Neural Network Better

### Fancier optimization

#### Problems with SGD

1. 不同的参数的下降速度可能差别很大，在敏感的维度上表现出较大梯度而在不敏感的维度上表现出较小的梯度
2. 在较为平坦的超曲面上梯度可能很小，从而使得梯度下降的速度很慢甚至停滞
3. 此外，SGD很容易陷入局部最小值，在高维空间中尤为如此，任何一个维度的前进都可能导致损失的增加
4. 噪声的存在也会显著影响梯度下降的性能

#### SGD + Momentum

保持一个不随时间变化的速度，并将梯度估计加上这个速度
$$
v_{t+1}=\rho v_t - \nabla f(x_t)\\x_{t+ 1} = x_t - \alpha v_{t+1}
$$
速度的存在使得梯度下降即使在梯度较小的时候也可以保持一定的速度，从而帮助算法越过局部最小值与鞍点。

```python
v = mu * v - learning_rate * dx
x += v
```

<img src="./CNN4Vision [6].assets/momentum.jpg" alt="查看源图像" style="zoom:50%;" />

#### Nesterov Momentum

Nesterov Momentum首先取得速度方向上步进一定距离，而后在该位置上取梯度。
$$
v_{t+1} = \rho v_t - \alpha\nabla f(x_t + \rho v_t)\\
x_{t+1} = x_t + v_{t+1}
$$
为了简化计算流程，做出以下的更改：
$$
\bar{x}_t = x_t + \rho v_t\\
v_{t+1} = \rho v_t - \alpha\nabla f(\bar{x}_t)\\
\begin{aligned}
\bar{x}_{t+1} &=\bar{x}_t - v_{t+1} \\
&=\bar{x}_t - \rho v_t + (1+ \rho)v_{t+1}\\
&=\bar{x}_t + v_{t+1} + \rho (v_{t+1} - v_t)
\end{aligned}
$$

<img src="./CNN4Vision [6].assets/image-20210413220316902.png" alt="image-20210413220316902" style="zoom: 60%;" />

#### AdaGrad

```python
grad_squared = 0
while True:
  dx = compute_gradient(x)
  grad_squared += dx * dx
  x -= learning_rate * dx / (np.sqrt(grad_squared) + 1e-7)
```

Adagrad求取梯度平方和的估计值，通过这样的方式，能够实现梯度的自适应缩放——加速小梯度维度的学习速度、降低大梯度维度的学习速度。此外随着迭代次数的增加，梯度平方和会逐渐增大，因此越接近最优参数，学习的速率会逐渐降低，从而更容易收敛（凸函数情形下如此，但是在非凸函数中这是复杂的问题）。

#### RMSProp

```python
grad_square = 0
while True:
  dx = compute_gradient(x)
  grad_squared = decay_rate * grad_squared + (1 - decay_rate) * dx * dx
  x -= learning_rate * dx / (np.sqrt(grad_squared) + 1e-7)
```

RMSProp梯度下降方法对梯度平方和施加了一个衰减率，这类似于动量的方法，只不过是对梯度平方和的估计添加动量。

#### :star:Adam

##### Original idea - combine Momenta and Adagrad/RMSProp

```python
first_moment = 0
second_moment = 0
while True:
  dx = compute_gradient(x)
  first_moment = beta1 * first_moment + (1 - beta1) * dx # Momentum
  second_moment = beta2 * second_moment + (1 - beta2) * dx * dx # AdaGrad/RMSProp
  x -= learning_rate * first_moment / (np.sqrt(second_moment) + 1e-7)
```

##### Full form

```python
first_moment = 0
second_moment = 0
for t in range(num_iterations):
  first_moment = beta1 * first_moment + (1 - beta1) * dx # Momentum
  second_moment = beta2 * second_moment + (1 - beta2) * dx * dx # AdaGrad/RMSProp
  first_unbias = first_moment / (1 - beta1 ** t) # bias correction
  second_unbias = second_moment / (1 - beta2 ** t) # bias correction
  x -= learning_rate * first_unbias / (np.sqrt(second_unbias) + 1e-7)
```

#### Learning rate dacay

学习率衰减也是提升优化效果的一个方法。

<img src="./CNN4Vision [6].assets/1*v8X32qzjxXowNuAvncl-ZA.png" alt="查看源图像" style="zoom:67%;" />

##### step decay - decay learning rate by half evey few epochs

##### exponential decay

$$
\alpha = \alpha_0 e^{-kt}
$$

##### 1/t decay

$$
\alpha = \alpha_0 / (1 + kt)
$$

#### First-Order optimization

1. 使用梯度来构建线性逼近
2. 步进来使得差距最小化
3. 上述所有的优化器都是一阶优化的

#### Second-Order optimization

1. 使用梯度和Hessian矩阵来构造二次逼近
2. 步进来到达最优处

$$
J(\theta)\approx J(\theta_0) + (\theta - \theta_0)^T\nabla_\theta J(\theta_0) + \frac{1}{2}(\theta -\theta_0)^T H(\theta - \theta_0)\\
\theta^* = \theta_0 - H^{-1}\nabla_\theta J(\theta_0)
$$

### Regularization

#### Dropout

在每一次前向传递时，以一定的概率随机设置**激活函数层的**某些神经元为0。其中抛弃神经元的概率为一个超参数，通常为0.5。

```python
def train_step(X):
  H1 = np.maximum(0, np.dot(W1, X) + b1)
  U1 = np.random.rand(*H1.shape) < p # first dropout mask
  H1 *= U1
  H2 = np.maximum(0, np.dot(W2, H1) + b2)
  U2 = np.random.rand(*H2.shape) < p # second dropout mask
  H2 *= U2
  out = np.dot(W3, H2) + b3
  
def predict(X):
  H1 = np.maximum(0, np.dot(W1, X) + b1) * p
  H2 = np.maximum(0, np.dot(W2, H1) + b2) * p
  out = np.dot(W3, H2) + b3
```

相较于batch normalization，dropout更加便于调整（超参数p）。此外，可以修改上述代码，使得在构造神经网络时更加简洁：

```python
def train_step(X):
  H1 = np.maximum(0, np.dot(W1, X) + b1)
  U1 = (np.random.rand(*H1.shape) < p) / p # first dropout mask
  H1 *= U1
  H2 = np.maximum(0, np.dot(W2, H1) + b2)
  U2 = (np.random.rand(*H2.shape) < p) / p # second dropout mask
  H2 *= U2
  out = np.dot(W3, H2) + b3
  
def predict(X):
  H1 = np.maximum(0, np.dot(W1, X) + b1)
  H2 = np.maximum(0, np.dot(W2, H1) + b2)
  out = np.dot(W3, H2) + b3
```

>**关于为什么在测试和预测时需要乘以概率$p$的问题：**
>
>因为我们在dropout层中以概率p抛弃神经元时，即(1-p)H个神经元的权重被置为0，最终输出的期望为px + (1 - p)0，为了在测试时或者预测时保持神经元的激活，必须要调整输出$x \rightarrow px$，使其保持一样的期望输出。

#### Data Augmentation

Horizontal Flip

Random crops and scales

Color Jitter 色彩抖动

#### DropConnect

随机将权重矩阵中的一些权重置为零

#### Fractional Max Pooling

在池化层操作时，随机池化当前正在池化的数据。

#### Stochastic Depth

训练时随机丢弃一些层，使用部分层，但在测试时使用所有层。

<img src="./CNN4Vision [6].assets/image-20210414103058565.png" alt="image-20210414103058565" style="zoom:50%;" />



