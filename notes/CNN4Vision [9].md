# CNN4Vision [9]

## Recurrent Neural Networks (RNN)

**Aim: Process Sequences**

<img src="./CNN4Vision [9].assets/Block-diagram-of-a-simple-RNN-that-unfolds-with-time-forming-a-chain-structure-A.png" alt="查看源图像" style="zoom:50%;" />
$$
h_t = f_W(h_{t-1},x_t)
$$
$h_t$ represents new state, $f_W$ some function with params $W$, $h_{t-1}$ old state and $x_t$ input vector at some time step.

### Vanilla RNN

$$
h_t = \tanh (W_{hh}h_{t-1} + W_{xh}x_t)\\
y_t = W_{hy} h_t
$$

<img src="./CNN4Vision [9].assets/91857201-8fd24280-eca2-11ea-848c-41e12102e4f8.png" alt="查看源图像" style="zoom: 67%;" />

在梯度下降的过程中，每一个block的权重都会计算各自的权重，并最终叠加到$W$中。

### Sequence to Sequence: Many-to-one & one-to-many

<img src="./CNN4Vision [9].assets/v2-278b5920ac2b4fc8c2319c90eaa7f9db_1200x500.jpg" alt="查看源图像" style="zoom: 33%;" />

### One-hot encoding

独热编码（一位有效编码）：使用N位状态寄存器对N个状态进行编码，每个状态都有其独立的寄存器位，并且任意时候只有一位有效。

### Truncated Backpropagation through time

RNN比较耗时且耗费内存资源，因为每做一次反向传播后，都需要从头开始进行前向传播。为了简化计算，提出了沿时间的截断反向传播方法（近似），其中在训练模型时，每一次都只前向计算100个单元。

<img src="./CNN4Vision [9].assets/truncated_backprop.png" alt="查看源图像" style="zoom:50%;" />

### Attention

### Multilayer RNNs

<img src="./CNN4Vision [9].assets/image_06_008.png" alt="查看源图像" style="zoom:50%;" />

### Vanilla RNN Gradient Flow

$$
\begin{aligned}
h_t &=\tanh (W_{hh}h_{t-1} + W_{xh} x_t)\\
&=\tanh \left(\begin{matrix}(W_hh & W_xh) \end{matrix}\left(\begin{matrix}(h_{t-1} \\ x_t) \end{matrix}\right)\right)\\
&= \tanh \left(W \left(\begin{matrix}h_{t-1} \\ x_t\end{matrix}\right)\right)
\end{aligned}
$$

<img src="./CNN4Vision [9].assets/assets%2F-LIA3amopGH9NC6Rf0mA%2F-M4bJ-IWAKzglR0XHFwU%2F-M4bJ2qhkeQ1-xH49LhY%2Frnn-gradient-flow.png" alt="查看源图像" style="zoom:50%;" />

若权重矩阵的最大奇异值大于1，有可能发生梯度爆炸，反之，若其奇异值小于1，则可能会产生梯度消失。

- 针对第一种情况，可以采用**Gradient Clipping**

```python
grad_norm = np.sum(grad * grad)
if grad_norm > threshold:
  grad *= (threshold / grad_norm)
```

- 对于第二种情况，则需要改变RNN的架构

### Long Short Term Memory

$$
\left(\begin{matrix}
i \\ f\\ o\\ g
\end{matrix}\right) = 
\left(\begin{matrix}
\sigma \\ \sigma \\ \sigma \\ \tanh
\end{matrix}\right) W 
\left(\begin{matrix}
h_{t-1} \\ x_t
\end{matrix}\right)\\
c_t = f\odot c_{t-1} + i\odot g\\
h_t = o\odot \tanh(c_t)
$$

在LSTM中有两个隐藏变量$h_t, c_t$，其中$c_t$为单元状态并且完全对

<img src="./CNN4Vision [9].assets/OIP.ceYsRNsrEG5aZYA3TQCbUAHaGk" alt="查看源图像"  />

- 遗忘门：是否忘记单元状态
- 输入门：是否写入/更新单元状态
- 门之门（Gate gate）：写入/更新多少信息
- 输出门：输出多少信息到下一个单元

LSTM的反向传播类似于ResNet，其通过单元状态能够实现梯度的快速传播。

