# CorneaNet
cornea segmentation

### 文件夹结构

```text
├── data.py					- 数据增强与处理文件
├── metric.py					- 评估指标（Jaccard/IoU score、F/Dice score）
├── loss.py					- 损失函数（crossentropy Losses、Jaccard Losses、Dice Losses）
├── model.py					- 模型结构
├── main.py					- 训练时运行的主文件
├── predict.py					- 调用模型进行预测
├── save_models
│   └── weights.264-0.1044.hdf5			- 训练出的最好模型
├── logs
│   └── events.out.tfevents.1558758583.cu02	- Tensorboard可视化
└── data
    ├── train					- 训练集24张
    │	├── image
    │	└── label
    ├── valid					- 训练集6张
    │	├── image
    │	└── label
    └── test					- 测试集20张（aug中扩张到200张最终用于测试）
 	├── image
 	├── label
 	└── aug
	    ├── predict				- 网络预测结果
	    └── gt				- 金标准
```
