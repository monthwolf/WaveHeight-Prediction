import torch 
from torch.autograd import Variable 
#from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt 
import numpy as np
from swhplot import caloss
from data_cache import xtrain, xtest, labeltrain, labeltest
  
# 设定随机数种子，保证每次运行结果的可重复性
torch.manual_seed(1) 
#mm = MinMaxScaler()
#y_train = torch.tensor(mm.fit_transform(labeltrain)).float()    #归一化处理

# 极限学习机模型参数
INPUT_SIZE = 2       # 输入特征维度（气压和风速）
HIDDEN_SIZE = 5      # 隐藏层节点数
OUTPUT_SIZE = 1      # 输出维度（海浪高度）

# 定义神经网络结构
model = torch.nn.Sequential( 
    torch.nn.Linear(INPUT_SIZE, HIDDEN_SIZE),   # 输入层到隐藏层连接
    torch.nn.Sigmoid(),                         # 隐藏层激活函数，ELM通常使用sigmoid
    torch.nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE)   # 隐藏层到输出层连接，无激活函数
)

# 训练ELM模型并保存参数
def train_and_save_elm():
    """
    训练极限学习机(ELM)模型并保存参数
    
    极限学习机是一种高效的单隐层前馈神经网络，其特点是：
    1. 输入权重和偏置随机初始化且无需调整
    2. 输出权重通过解析解(Moore-Penrose广义逆)直接计算
    3. 训练速度极快，无需迭代优化
    """
    # 定义输入层到隐藏层的网络，仅用于计算隐藏层输出
    input_hidden_layer = torch.nn.Sequential( 
        torch.nn.Linear(INPUT_SIZE, HIDDEN_SIZE),   # 输入层到隐藏层连接
        torch.nn.Sigmoid()                          # 隐藏层激活函数
    )
    
    # 计算隐藏层输出矩阵H
    hidden_output = input_hidden_layer(xtrain).squeeze(-1)
    
    # ELM的核心：利用Moore-Penrose广义逆求解输出权重
    # 计算隐层输出矩阵的广义逆矩阵 H†
    H_inv = np.linalg.pinv(hidden_output.data.numpy().reshape(len(labeltrain), HIDDEN_SIZE))
    # 目标输出矩阵T
    T = labeltrain.data.numpy().reshape(len(labeltrain))
    # 计算输出权重 β = H†T
    output_weights = np.dot(H_inv, T)
    output_weights_tensor = torch.tensor(output_weights).float()
    
    # 将权重应用到完整网络
    model.state_dict()['0.weight'].copy_(input_hidden_layer.state_dict()['0.weight'])  # 复制输入层权重
    model.state_dict()['0.bias'].copy_(input_hidden_layer.state_dict()['0.bias'])      # 复制隐藏层偏置
    model.state_dict()['2.weight'].copy_(output_weights_tensor)                         # 设置计算得到的输出层权重
    model.state_dict()['2.bias'].copy_(torch.tensor(0))                                 # 输出层偏置设为0
    
    # 保存ELM神经网络参数
    torch.save(model.state_dict(), 'elmnet_params.pkl')
    print("ELM模型训练完成并已保存")
    return model
  
# 加载训练好的模型参数并进行预测和评估
def evaluate_elm_model():
    """
    载入已训练好的ELM模型参数，并在测试集和训练集上进行预测和评估
    
    评估模型在测试集和训练集上的性能，计算各种评价指标，并可视化预测结果
    """
    # 载入保存的神经网络参数
    model.load_state_dict(torch.load('elmnet_params.pkl'))
    loss_function = torch.nn.MSELoss()
    
    # 在测试集上进行预测和评估
    print("\n===== 测试集评估结果 =====")
    test_predictions = model(xtest).squeeze(-1)
    caloss(test_predictions, labeltest)  # 计算评价指标
    
    # 可视化测试集预测结果
    plot_prediction_results(labeltest, test_predictions, "ELM Test Set Prediction")
    
    # 在训练集上进行预测和评估
    print("\n===== 训练集评估结果 =====")
    train_predictions = model(xtrain).squeeze(-1)
    caloss(train_predictions, labeltrain)  # 计算评价指标
    
    # 可视化训练集预测结果
    plot_prediction_results(labeltrain, train_predictions, "ELM Training Set Prediction")

# 辅助函数：绘制预测结果可视化图
def plot_prediction_results(true_values, predicted_values, title):
    """
    绘制预测值与真实值的散点图和拟合线
    
    参数:
        true_values: 真实标签值
        predicted_values: 模型预测值
        title: 图表标题
    """
    plt.figure(figsize=(10, 8))
    plt.title(title)
    
    # 计算线性回归拟合线
    true_np = true_values.squeeze(-1).data.numpy()
    pred_np = predicted_values.squeeze(-1).data.numpy()
    coefficients = np.polyfit(true_np, pred_np, 1)
    polynomial = np.poly1d(coefficients, variable='x')
    
    # 计算R方（决定系数）
    loss_function = torch.nn.MSELoss()
    rsquare = 1 - loss_function(true_values, predicted_values).item() / np.var(true_values.data.numpy())
    
    # 绘制散点图
    plt.scatter(true_values.data.numpy(), predicted_values.data.numpy(), alpha=0.6)
    plt.xlabel('True Wave Height')
    plt.ylabel('Predicted Wave Height')
    
    # 绘制拟合直线
    plt.plot(true_values.data.numpy(), polynomial(true_values.data.numpy()), 'r--', linewidth=2)
    
    # 添加拟合方程和R方值文本框
    equation_text = f'y = {coefficients[0]:.4f}x + {coefficients[1]:.4f}\nR² = {rsquare:.4f}'
    plt.text(0.05, 0.95, equation_text, transform=plt.gca().transAxes,
             verticalalignment="top", bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

# 程序执行入口
if __name__ == "__main__":
    print("开始训练极限学习机模型...")
    model = train_and_save_elm()
    print("\n开始评估极限学习机模型...")
    evaluate_elm_model()