NSL-KDD 网络入侵检测系统 - GAT 模型实现
基于图注意力网络 (GAT) 的 NSL-KDD 数据集网络入侵检测系统，通过构建 KNN 图结构捕捉网络流量样本间的关联关系，提升入侵检测性能。
项目结构
plaintext
├── graph_construction.py   # KNN图构建模块
├── main.py                 # 主程序入口
├── Build_GAT_model.py      # GAT模型定义
├── train_evaluate.py       # 训练与评估函数
├── data_preprocessing.py   # 数据预处理模块
├── load_model.py           # 模型加载工具
└── nsl_kdd_gat_best.pth    # 训练好的模型(运行后生成)
环境依赖
Python 3.8+   PyTorch 1.10+  PyTorch Geometric 2.0+  scikit-learn  pandas  numpy
安装依赖：
pip install torch torch-geometric scikit-learn pandas numpy
数据集
使用 NSL-KDD 数据集，包含：
KDDTrain+.txt：训练集
KDDTest+.txt：测试集
默认路径设置为：D:\code\pythonProject2\NSL-KDD-DataSet-master\，可在data_preprocessing.py中修改load_data函数的文件路径。
主要功能
1.数据预处理：
分层抽样（默认 50%）  标签编码（22 类网络攻击类型）  分类特征 One-Hot 编码  数值特征标准化 缺失值处理
2.图结构构建：
基于 KNN 算法构建图（k=5）  生成双向边关系  转换为 PyTorch Geometric 数据格式
3.GAT 模型：
输入特征维度：140 维 隐藏层维度：128 维  网络层数：3 层  注意力头数：6 头  包含批归一化和 dropout
4.训练配置：
优化器：Adam（学习率 0.001） 损失函数：带类别权重的交叉熵  早停机制（patience=50） 最大训练轮次：100
使用方法
一.训练模型
直接运行主程序
程序执行流程：
1.加载并抽样数据 2.数据预处理（特征工程） 3.构建 KNN 图结构 4.初始化 GAT 模型 5.带早停机制的模型训练 6.模型评估（准确率、F1 分数、分类报告） 7.保存最佳模型
二.加载已训练模型
使用load_model.py加载保存的模型： 运行python load_model.py
模型性能
模型在测试集上的典型性能：
准确率：约 55-65%
加权 F1 分数：约 0.55-0.65
详细分类报告和混淆矩阵会在训练结束后自动打印。
核心参数配置
可在main.py中调整的核心参数：
sample_fraction：数据抽样比例（默认 0.5）
k：KNN 图的近邻数量（默认 5）
模型配置：隐藏层维度、层数、注意力头数
训练参数：学习率、早停耐心值、最大轮次
备注
模型默认使用 CPU 训练，可在main.py中修改device参数启用GPU
分类报告中可能存在部分类别 F1 分数较低，因数据集中存在样本量极少的类别
模型保存为nsl_kdd_gat_best.pth，包含模型权重、输入维度和标签编码器信息