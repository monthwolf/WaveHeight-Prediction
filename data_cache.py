import torch
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler 
from ncread import msl, uv10, swh

# 数据处理模块：负责对从NetCDF文件读取的原始数据进行预处理，准备模型训练所需的输入数据和标签

# 设置数据集划分参数
TRAIN_SAMPLES = 7320  # 训练集样本数
TEST_SAMPLES = 1464   # 测试集样本数
TOTAL_SAMPLES = TRAIN_SAMPLES + TEST_SAMPLES  # 总样本数

# 创建训练集和测试集标签张量（初始化）
train_labels = torch.unsqueeze(torch.linspace(-1, 1, TRAIN_SAMPLES), dim=1)  # 创建训练集标签占位张量
test_labels = torch.unsqueeze(torch.linspace(-1, 1, TEST_SAMPLES), dim=1)    # 创建测试集标签占位张量

# 创建存储特征的列表
train_features = []  # 训练集特征列表
test_features = []   # 测试集特征列表

# 划分数据集：从ncread.py导入的数据中提取训练集和测试集
train_pressure = msl[0:TRAIN_SAMPLES]      # 训练集气压数据
test_pressure = msl[TRAIN_SAMPLES:TOTAL_SAMPLES]    # 测试集气压数据
train_wind = uv10[0:TRAIN_SAMPLES]         # 训练集风速数据
test_wind = uv10[TRAIN_SAMPLES:TOTAL_SAMPLES]       # 测试集风速数据

# 填充训练集数据 - 训练标签为前TRAIN_SAMPLES个海浪高度值，特征为气压和风速
for i in range(TRAIN_SAMPLES):
  train_labels[i] = swh[i]                            # 设置训练集标签为实际海浪高度值
  train_features.append([train_pressure[i], train_wind[i]])  # 设置训练集特征为[气压,风速]

# 填充测试集数据 - 测试标签为后TEST_SAMPLES个海浪高度值，特征为气压和风速
for i in range(TEST_SAMPLES):
  test_labels[i] = swh[i+TRAIN_SAMPLES]                        # 设置测试集标签为实际海浪高度值
  test_features.append([test_pressure[i], test_wind[i]])       # 设置测试集特征为[气压,风速]

# 特征标准化处理：使用MinMaxScaler将特征值标准化到[0,1]范围内
scaler = MinMaxScaler()
train_features_scaled = scaler.fit_transform(train_features)  # 对训练集特征进行拟合和变换
test_features_scaled = scaler.transform(test_features)        # 对测试集特征进行变换（使用训练集的缩放参数）

# 转换为PyTorch张量并调整维度为[样本数, 1, 特征数]形式，用于后续模型处理
xtrain = torch.unsqueeze(torch.tensor(train_features_scaled).float(), dim=1)
xtest = torch.unsqueeze(torch.tensor(test_features_scaled).float(), dim=1)

# 创建PyTorch变量，设置requires_grad=True以便进行梯度计算
xtrain = Variable(xtrain, requires_grad=True)
xtest = Variable(xtest, requires_grad=True)

# 保持原变量名，兼容现有代码
labeltrain = train_labels
labeltest = test_labels

#ytrain = y_train + 0.2*torch.rand(y_train.size())     #加入噪声提高鲁棒性
#ytest = y_test + 0.2*torch.rand(y_test.size()) 