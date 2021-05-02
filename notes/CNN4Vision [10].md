# CNN4Vision [10]

## Classification and Localization

<img src="./CNN4Vision [10].assets/classification-localization.png" alt="classification-localization"  />

### Objection Detection as Classification: Sliding Window

将CNN应用于一幅图像中的不同块，其会将这些块分为背景抑或是对象。但是存在严重的问题，块的划分严重影响检测的结果，因此CNN需要检测大量的不同位置与尺度，从而带来非常大的计算代价。

### Region Proposals

在图像中寻找团块（blob），其中可能包含对象，然后在备选团块中选择最佳的存在目标对象的。（Refer to **R-CNN**）

<img src="./CNN4Vision [10].assets/Selective-search-for-region-proposal-generation-Source-378.png" alt="See the source image" style="zoom:67%;" />

#### R-CNN

<img src="./CNN4Vision [10].assets/R832959fe24c00e2abd65e5fe978975e8.jpeg" alt="See the source image" style="zoom:50%;" />

#### Faster R-CNN

<img src="./CNN4Vision [10].assets/Rc1fd56906727c3d9c0916391d4162bec.jpeg" alt="See the source image" style="zoom: 33%;" />

<img src="./CNN4Vision [10].assets/R76b85491ecba59b1e5196b9d64d113f7.png" alt="See the source image" style="zoom: 33%;" />

#### Mask R-CNN

<img src="./CNN4Vision [10].assets/R9effcdd23972199e09c4a5b0e4bfb095.png" alt="See the source image"  />

Mask R-CNN本质上是对每一类别都预测出一个边界框，这个过程类似于图像分割。Mask R-CNN同时也可以实现姿态的估计。

### Detection without Proposals: YOLO / SSD

这类方法为前馈模型，其将检测+定位稳定作为回归问题处理，借助大型卷积神经网络一次完成。在这一类模型中，首先要预测边界框的偏移（边界框与目标物体位置的偏差），而后需要预测目标对应的分数（某个类别出现在边框中的概率）。

<img src="./CNN4Vision [10].assets/R817a4029c54d9eb969612503b51d7e27.png" alt="See the source image" style="zoom:67%;" />

