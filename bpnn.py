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
#ytrain = torch.tensor(mm.fit_transform(labeltrain)).float()    #归一化处理

# BP神经网络的配置参数
INPUT_SIZE = 2        # 输入特征维度（气压和风速）
HIDDEN_SIZE = 5       # 隐藏层节点数
OUTPUT_SIZE = 1       # 输出维度（海浪高度）
LEARNING_RATE = 0.3   # 学习率
MAX_EPOCHS = 500      # 最大迭代次数
EARLY_STOP_LOSS = 0.04  # 早停阈值

# 定义和训练BP神经网络模型
def train_and_save_model(): 
    """
    定义、训练BP神经网络模型并保存参数
    
    BP神经网络是最基础的前馈神经网络，通过梯度下降法训练，用于本项目的海浪高度预测
    本函数构建一个2-5-1结构的BP神经网络，使用随机梯度下降算法训练，并记录训练过程中的误差变化
    """
    # 构建神经网络结构：输入层(2) -> 隐藏层(5) -> 输出层(1)
    model = torch.nn.Sequential( 
        torch.nn.Linear(INPUT_SIZE, HIDDEN_SIZE),    # 输入层到隐藏层连接
        torch.nn.ReLU(),                             # 隐藏层激活函数，采用ReLU提高训练效率
        torch.nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE),   # 隐藏层到输出层连接
        torch.nn.Softplus()                          # 输出层激活函数，保证输出为正值（海浪高度）
    ) 
    
    # 定义优化器和损失函数  
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)  # 随机梯度下降优化器
    loss_function = torch.nn.MSELoss()                                 # 均方误差损失函数
    loss_history = []                                                  # 记录训练过程中的损失值
    
    # 训练循环
    for epoch in range(MAX_EPOCHS): 
        # 前向传播，计算预测值
        predictions = model(xtrain)
        predictions = predictions.squeeze(-1)  # 调整预测值维度
        
        # 计算损失
        loss = loss_function(predictions, labeltrain)
        loss_history.append(loss.item())  # 记录当前损失值
        
        # 当损失值小于阈值时提前结束训练
        if loss.item() <= EARLY_STOP_LOSS:
            print(f"Early stopping at epoch {epoch} with loss {loss.item():.4f}")
            break
        
        # 反向传播和参数更新
        optimizer.zero_grad()  # 梯度清零
        loss.backward()        # 反向传播，计算梯度
        optimizer.step()       # 更新参数
    
    # 绘制训练过程中的误差曲线 
    plt.figure(figsize=(8, 6))
    plt.title('BP Neural Network Training Loss')  
    plt.plot(loss_history) 
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.grid(True)
    plt.show()
    
    # 保存训练好的神经网络参数
    torch.save(model.state_dict(), 'net_params.pkl')
    return model
    
# 加载训练好的模型参数并进行预测和评估
def evaluate_model(): 
    """
    载入已训练好的BP神经网络模型参数，并在测试集和训练集上进行预测和评估
    
    加载之前保存的模型参数，在测试集上进行预测，并通过多种评价指标评估模型性能，
    包括MSE, R方等，同时绘制预测值与真实值的散点图和拟合线，直观展示预测效果
    """
    # 构建与训练时相同的神经网络结构 
    model = torch.nn.Sequential( 
        torch.nn.Linear(INPUT_SIZE, HIDDEN_SIZE), 
        torch.nn.ReLU(), 
        torch.nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE), 
        torch.nn.Softplus()
    ) 
    
    # 载入神经网络的模型参数 
    model.load_state_dict(torch.load('net_params.pkl')) 
    loss_function = torch.nn.MSELoss()
    
    # 测试集评估
    print("\n===== 测试集评估结果 =====")
    test_predictions = model(xtest).squeeze(-1)
    caloss(test_predictions, labeltest)  # 计算评价指标
    
    # 绘制测试集预测结果散点图与拟合线
    plot_prediction_vs_truth(labeltest, test_predictions, "Test Set Prediction")
    
    # 训练集评估
    print("\n===== 训练集评估结果 =====")
    train_predictions = model(xtrain).squeeze(-1)
    caloss(train_predictions, labeltrain)  # 计算评价指标
    
    # 绘制训练集预测结果散点图与拟合线
    plot_prediction_vs_truth(labeltrain, train_predictions, "Training Set Prediction")

# 辅助函数：绘制预测值与真实值的散点图和拟合线
def plot_prediction_vs_truth(truth, prediction, title):
    """
    绘制预测值与真实值的散点图和拟合线，并显示拟合方程和R方值
    
    参数:
        truth: 真实标签
        prediction: 模型预测值
        title: 图表标题
    """
    plt.figure(figsize=(10, 8))
    plt.title(title) 
    
    # 计算线性回归拟合线
    truth_np = truth.squeeze(-1).data.numpy()
    pred_np = prediction.squeeze(-1).data.numpy()
    param = np.polyfit(truth_np, pred_np, 1)    # 线性回归方程拟合
    p = np.poly1d(param, variable='x')          # 获取方程式
    
    # 计算R方（决定系数）
    loss_function = torch.nn.MSELoss()
    rsquare = 1 - loss_function(truth, prediction).item() / np.var(truth.data.numpy())
    
    # 绘制散点图
    plt.scatter(truth.data.numpy(), prediction.data.numpy(), alpha=0.6) 
    plt.xlabel('True Wave Height')
    plt.ylabel('Predicted Wave Height')
    
    # 绘制拟合直线
    plt.plot(truth.data.numpy(), p(truth.data.numpy()), 'r--', linewidth=2)  
    
    # 添加拟合方程和R方值
    equation_text = f'y = {param[0]:.4f}x + {param[1]:.4f}\nR² = {rsquare:.4f}'
    plt.text(0.05, 0.95, equation_text, transform=plt.gca().transAxes,
             verticalalignment="top", bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

# 程序执行入口：先训练神经网络，然后测试评估
if __name__ == "__main__":
    print("开始训练BP神经网络模型...")
    model = train_and_save_model()
    print("\n开始评估BP神经网络模型...")
    evaluate_model()