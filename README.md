# CDNet

CDNet 的 PyTorch 训练代码.

相比于测试代码，这里模型内部的一些参数名称做了调整，与测试代码相比会有不同。非常抱歉，由于个人时间原因，训练代码并没有完全注释，并且也没有对这部分内容进行调试，直接运行可能会报错。但考虑到有些同学可能需要确定模型的超参数、训练过程以及使用的损失函数，所以放出目前的版本。

## Saliency maps

我们提供了两个预训练 CDNet 在 8 个测试集:
NJU, NLPR, LSFD, DES, STERE, SSD, SIP, DUT_test 上的测试结果。

**CDNet_results.zip**:
在 NJU+NLPR 上进行训练的 CDNet 的测试结果。其中，后缀为 "_pred_0.png" 表示显著图，后缀为 "_pred_1.png" 表示预测出的深度图，后缀为 "_pred_2.png" 表示预测出的边缘。
[BaiduYun](https://pan.baidu.com/s/17g6M_WPTu7lGhdOKszFrlg) (提取码：thmr)

**CDNet_results_2.zip**:
在 NJU+NLPR+DUT 上进行训练的 CDNet 的测试结果。其中，无后缀表示显著图，后缀为 "_pred_1.png" 表示预测出的深度图，后缀为 "_pred_2.png" 表示预测出的边缘。
[BaiduYun](https://pan.baidu.com/s/1IhsLbFh6J7FArL6QGxt8jA) (提取码：0d1x)
