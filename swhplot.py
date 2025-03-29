import matplotlib.pyplot as plt
import torch
import numpy as np
from data_cache import swh

# 绘制海浪高度时间序列函数
def plot_wave_height_timeseries():
    """
    绘制海浪高度时间序列图
    
    该函数绘制训练集和测试集的海浪高度随时间变化的曲线图，
    用于直观展示海浪高度的变化趋势和时间特性
    """
    # 绘制训练集海浪高度时间序列
    plt.figure(figsize=(12, 6))
    plt.title('Training Set Wave Height Time Series', fontsize=14)  
    plt.plot(swh[0:16104], 'b-', linewidth=1.2) 
    plt.xlabel('Time Index', fontsize=12)    # X轴表示时间序列索引
    plt.ylabel('Wave Height (m)', fontsize=12)  # Y轴表示海浪高度(m)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    
    # 绘制测试集海浪高度时间序列
    plt.figure(figsize=(12, 6))
    plt.title('Test Set Wave Height Time Series', fontsize=14)  
    plt.plot(swh[16104:17568], 'g-', linewidth=1.2) 
    plt.xlabel('Time Index', fontsize=12)    # X轴表示时间序列索引
    plt.ylabel('Wave Height (m)', fontsize=12)  # Y轴表示海浪高度(m)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# 计算多种评价指标函数
def calculate_metrics(predictions, ground_truth):
    """
    计算模型预测结果的多种评价指标
    
    参数:
        predictions: 模型预测值张量
        ground_truth: 真实标签值张量
    
    返回:
        metrics_dict: 包含各评价指标的字典
    """
    # 将输入转换为numpy数组以便计算
    pred_np = predictions.data.numpy()
    truth_np = ground_truth.data.numpy()
    
    # 计算MAE(平均绝对误差)：衡量预测值与真实值的平均绝对差异
    mae = torch.nn.L1Loss()(predictions, ground_truth).item()
    
    # 计算MSE(均方误差)：衡量预测值与真实值差异的平方和的平均值
    mse = torch.nn.MSELoss()(predictions, ground_truth).item()
    
    # 计算RMSE(均方根误差)：MSE的平方根，能够反映预测值与真实值的平均偏差
    rmse = np.sqrt(mse)
    
    # 计算MAPE(平均绝对百分比误差)：衡量预测值相对于真实值的平均绝对百分比误差
    mape = np.mean(np.abs((pred_np - truth_np) / truth_np)) * 100
    
    # 计算NSE(Nash-Sutcliffe效率系数)：评价模型预测效果相对于简单平均值模型的改进程度
    # NSE = 1表示完美预测，NSE = 0表示预测效果与使用观测均值一样好，NSE < 0表示均值模型更好
    pred_mean = torch.full_like(predictions, torch.mean(predictions).item())
    nse = 1 - (mse / torch.nn.MSELoss()(pred_mean, ground_truth).item())
    
    # 创建包含所有指标的字典
    metrics_dict = {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "MAPE": mape,
        "NSE": nse
    }
    
    # 打印评价指标
    print("\n模型评估指标:")
    print(f"MAE (平均绝对误差): {mae:.4f}")
    print(f"RMSE (均方根误差): {rmse:.4f}")
    print(f"MAPE (平均绝对百分比误差): {mape:.2f}%")
    print(f"NSE (Nash-Sutcliffe效率系数): {nse:.4f}")
    
    return metrics_dict

# 兼容旧代码的评价指标函数
def caloss(predictions, ground_truth):
    """
    计算并打印模型评价指标（兼容旧代码）
    
    参数:
        predictions: 模型预测值
        ground_truth: 真实标签值
    """
    return calculate_metrics(predictions, ground_truth)

# 多模型预测结果对比绘图函数
def compare_model_predictions(test_features, test_labels, model_area="default"):
    """
    对比不同模型在同一测试集上的预测性能
    
    参数:
        test_features: 测试集特征
        test_labels: 测试集标签
        model_area: 标识字符串，用于区分不同海域('bohai'表示渤海，'yellow'表示黄海)
    
    功能:
        加载不同模型(BPNN、IPSO-BP、IPSO-ELM)预测结果，并在同一图上绘制对比图
    """
    # 将输入数据转换为PyTorch张量并调整维度
    test_labels_tensor = torch.unsqueeze(torch.tensor(test_labels).float(), dim=1)
    test_features_tensor = torch.unsqueeze(torch.tensor(test_features).float(), dim=1)
    test_features_variable = torch.autograd.Variable(test_features_tensor, requires_grad=True)

    # 定义BP神经网络结构
    bp_model = torch.nn.Sequential( 
        torch.nn.Linear(2, 5), 
        torch.nn.ReLU(), 
        torch.nn.Linear(5, 1),
        torch.nn.Softplus()
    )
    
    # 定义ELM神经网络结构
    elm_model = torch.nn.Sequential( 
        torch.nn.Linear(2, 5), 
        torch.nn.Sigmoid(), 
        torch.nn.Linear(5, 1)
    )
    
    # 加载各个模型参数并进行预测
    bp_model.load_state_dict(torch.load('net_params.pkl'))             # 加载标准BP神经网络参数
    bp_predictions = bp_model(test_features_variable).squeeze(-1)       # BP神经网络预测
    
    bp_model.load_state_dict(torch.load('ipsoBP_params.pkl'))          # 加载IPSO优化的BP神经网络参数
    ipso_bp_predictions = bp_model(test_features_variable).squeeze(-1)  # IPSO-BP神经网络预测
    
    elm_model.load_state_dict(torch.load('ipsoELM_params.pkl'))        # 加载IPSO优化的ELM神经网络参数
    ipso_elm_predictions = elm_model(test_features_variable).squeeze(-1)  # IPSO-ELM神经网络预测
    
    # 绘制预测结果对比图
    plt.figure(figsize=(14, 8))
    
    # 设置标题和坐标轴标签
    if model_area == "bohai":
        plt.title('Bohai Sea Wave Height Prediction Comparison', fontsize=16)
    elif model_area == "yellow":
        plt.title('Yellow Sea Wave Height Prediction Comparison', fontsize=16)
    else:
        plt.title('Wave Height Prediction Model Comparison', fontsize=16)
        
    plt.xlabel('Time Index', fontsize=14)       # X轴表示时间序列索引
    plt.ylabel('Wave Height (m)', fontsize=14)  # Y轴表示预测的海浪高度
    
    # 绘制真实值和各模型预测值曲线
    plt.plot(test_labels_tensor.data.numpy(), 'g', lw=2, label='Ground Truth')       # 绘制真实值曲线(绿色)
    plt.plot(bp_predictions.data.numpy(), 'r', lw=1.5, label='BPNN')                 # 绘制BP神经网络预测值曲线(红色)
    plt.plot(ipso_bp_predictions.data.numpy(), 'b', lw=1.5, label='IPSO-BPNN')       # 绘制IPSO-BP预测值曲线(蓝色)
    plt.plot(ipso_elm_predictions.data.numpy(), 'k', lw=1.5, label='IPSO-ELM')       # 绘制IPSO-ELM预测值曲线(黑色)
    
    # 添加图例和网格线
    plt.legend(loc='upper left', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    # 计算并显示各模型的评价指标
    print("\n===== BP神经网络评价指标 =====")
    bp_metrics = calculate_metrics(bp_predictions, test_labels_tensor)
    
    print("\n===== IPSO-BP神经网络评价指标 =====")
    ipso_bp_metrics = calculate_metrics(ipso_bp_predictions, test_labels_tensor)
    
    print("\n===== IPSO-ELM评价指标 =====")
    ipso_elm_metrics = calculate_metrics(ipso_elm_predictions, test_labels_tensor)
    
    plt.show()
    
    # 返回各模型的评价指标，方便进一步分析
    return {
        "BP": bp_metrics,
        "IPSO-BP": ipso_bp_metrics,
        "IPSO-ELM": ipso_elm_metrics
    }

# 主函数：用于直接运行脚本时展示图表
if __name__ == "__main__":
    print("绘制海浪高度时间序列图...")
    plot_wave_height_timeseries()
    
    # 注释掉的特定时间点海浪高度预测结果比较代码（保留作为示例）
    """
    # 提取每日8:00的数据点进行预测对比
    # 黄海区域
    yellow_sea_test_labels = []
    yellow_sea_test_features = []
    for i in range(30):
        yellow_sea_test_labels.append(labeltest[i*48])
        yellow_sea_test_features.append(x_test[i*48])
    compare_model_predictions(yellow_sea_test_features, yellow_sea_test_labels, "yellow")
    
    # 渤海区域
    bohai_test_labels = []
    bohai_test_features = []
    for i in range(30):
        bohai_test_labels.append(labeltest[i*48+6])
        bohai_test_features.append(x_test[i*48+6])
    compare_model_predictions(bohai_test_features, bohai_test_labels, "bohai")
    """

