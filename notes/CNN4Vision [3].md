# CNN4Vision [3]

## Backpropagation and Neural Networks

### Computational graphs

使用计算图可以表示任意的函数，其中的节点表示某一个计算过程。

<img src="./CNN4Vision [3].assets/1*APizoaj5X1Hh-JaujnM6Kw.png" alt="查看源图像"  />

### Backpropagation

#### Forward

计算每个操作的结果，并将其存储为中间变量，以便后续进行反向更新权重。

#### Backward

反向传播梯度更新算法利用神经网络的链式法则逐步对从网络输出端向输入端方向的所有参数进行更新。

<img src="./CNN4Vision [3].assets/backprop-template.png" alt="backpropagation" style="zoom: 33%;" />

 反向传播将复杂的梯度计算分解为简单的计算（如乘法、加法等），从而简化了其梯度的解析式。

<img src="./CNN4Vision [3].assets/backpropagation.jpg" alt="backpropagation" style="zoom: 100%;" />

- **add** gate -> gradient distributor 只是传递梯度，本身不改变梯度
- **max** gate -> gradient router 只对具有较大值的参数进行更新（梯度为0或1）
- **mul** gate -> gradient switcher 对梯度的大小进行缩放

#### Vectorized computation

<img src="./CNN4Vision [3].assets/OIP.N1Jqa6IzXUfuiXX12ALKnAHaDr" alt="查看源图像" style="zoom:150%;" />

### Neural Networks

- 2-layer neural networks

$$
f=W_2\max(0, W_1x)
$$

- 3-layer neural networks

$$
f=W_3\max(0, W_2\max(0, W_1x)))
$$

#### Activation Functions

- Sigmoid
- tanh
- ReLU
- Leaky ReLU
- Maxout
- ELU

![查看源图像](./CNN4Vision [3].assets/1*ZafDv3VUm60Eh10OeJu1vw.png)

#### Achitectures

