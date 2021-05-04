# CNN4Vision [11]

## Segmentation, Localization and Detection

###  Semantic segmentation

<img src="./CNN4Vision [11].assets/Example-of-2D-semantic-segmentation-Top-input-image-Bottom-prediction.png" alt="查看源图像" style="zoom:50%;" />

#### #Idea 1: Sliding window

将图像划分为小块，并对其进行分类。其缺点在于计算代价过大。

<img src="./CNN4Vision [11].assets/OIP-20210504104020660.VziQiG6VbVtPhPk6JDCRlQHaDF" alt="查看源图像"  />

#### #Idea 2: Fully Convolutional

采用全卷积神经网络对每个像素计算分数并进行分类。其中利用补零的方式来保证经过卷积处理后与原图像一致的尺寸。缺点仍是计算量过大，并且数据集获取比较困难。

<img src="./CNN4Vision [11].assets/0186005ef38b4c77ceb7cccd8f904853.png" alt="查看源图像" style="zoom:50%;" />

一个改进的版本是在神经网络中进行下采样，而后再进行上采样。下采样的手段比较简单，之间利用池化（Max Pooling、Average Pooling），而上采样采用的是反池化或者去池化（如下图所示）。

<img src="./CNN4Vision [11].assets/semantic_segmentation_fully_convolutional_sampling.png" alt="查看源图像" style="zoom:50%;" />

<img src="./CNN4Vision [11].assets/Illustration-of-the-effects-of-the-maxpooling-and-unpooling-operations-on-a-4-A-4-feature.png" alt="查看源图像"  />

##### transpose convolution 反卷积/转置卷积

<img src="./CNN4Vision [11].assets/semantic_segmentation_transpose_convolution_1d_example.png" alt="查看源图像" style="zoom: 33%;" />

