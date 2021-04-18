# CNN4Vision [7]

## Hardwares and Softwares for DL

### CPU vs GPU

<img src="https://i2.wp.com/www10.mcadcafe.com/blogs/jeffrowe/files/2017/03/CPU-and-GPU.png?resize=925%2C625&ssl=1" alt="查看源图像" style="zoom:75%;" />

### DL framework

#### PyTorch

1. **Tensor**: Imperative ndarray but runs on GPU
2. **Variable**: Node in a computational graph; stores data and gradient
3. **Module**: A neural network layer; may store state or learnable weights

PyTorch可以自定义新的autograd函数

```python
class ReLU(torch.autograd.Function):
  def forward(self, x):
    self.save_for_backward(x)
    return x.clamp(min=0)
  
  def backward(self, grad_y):
    x, = self.saved_tensors
    grad_input = grad_y.clone()
    grad_input[x < 0] = 0
    return grad_input
```

PyTorch对较低层次的API进行了高级封装，可以通过nn模块来构建所需要的神经网络结构

```python
import torch
from torch.autograd import Variable

N, D_in, H, D_out = 64, 1000, 100, 10
x = Variable(torch.randn(N, D_in))
y = Variable(torch.randn(N, D_out), requires_grad=False)

model = torch.nn.Sequential([
  torch.nn.Linear(D_in, H),
  torch.nn.ReLU(),
  torch.nn.Linear(H, D_out),
  torch.nn.MSELoss(size_average=False)
])

learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for t in range(500):
  y_pred = model(x)
  loss = loss_fn(y_pred, y)
  
  optimizer.zero_grad()
  loss.backward()
  
  optimizer.step()
```

PyTorch同样也可以自定义新的模块

```python
class TwoLayerNet(torch.nn.Module):
  def __init__(self, D_in, H, D_out):
    super(TwoLayerNet, self).__init__()
    self.linear1 = torch.nn.Linear(D_in, H)
    self.linear2 = torch.nn.Linear(H, D_out)
  
  def forward(self, x):
    h_relu = self.linear1(x).clamp(min=0)
    y_pred = self.linear2(h_relu)
    return y_pred
```

PyTorch中的分批数据读取——dataloader，可以自动执行分批、打乱等处理

#### Static vs Dynamic Graphs

- 静态图：只建立一次，然后运行许多次
  - 更加适合部署在生产环境
  - 不太pythonic，类似于构建了新的语言，需要使用其规定的范式来构建网络

- 动态图：每一次迭代都建立新的图
  - 更加灵活，可以直接使用python代码来构建比较复杂的结构（例如条件语句）



