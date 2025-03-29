import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from data_cache import *



# 多模型预测结果对比绘图函数
def swhfullin(xtest, ytest, flag):
    """
    对比不同模型在同一测试集上的预测性能
    
    参数:
        xtest: 测试集特征
        ytest: 测试集标签
        flag: 标识字符串，用于区分不同海域('hh'表示黄海，其他值表示渤海)
    
    功能:
        加载不同模型(BPNN、IPSO-BP、IPSO-ELM)预测结果，并在同一图上绘制对比图
    """
    # 将输入数据转换为PyTorch张量并调整维度
    ytest = torch.unsqueeze(torch.tensor(ytest).float(), dim=1)   # 升维
    xtest = torch.unsqueeze(torch.tensor(xtest).float(), dim=1)
    xtest = Variable(xtest, requires_grad=True)

    # 定义BP神经网络结构
    bpnet = torch.nn.Sequential( 
    torch.nn.Linear(2, 5), 
    torch.nn.ReLU(), 
    torch.nn.Linear(5, 1),
    torch.nn.Softplus()
    )
    
    # 定义ELM神经网络结构
    elmnet = torch.nn.Sequential( 
    torch.nn.Linear(2, 5), 
    torch.nn.Sigmoid(), 
    torch.nn.Linear(5, 1)
    )
    
    # 加载各个模型参数并进行预测
    bpnet.load_state_dict(torch.load('net_params.pkl'))         # 加载标准BP神经网络参数
    prediction1 = bpnet(xtest).squeeze(-1)                      # BP神经网络预测
    #elmnet.load_state_dict(torch.load('elmnet_params.pkl'))
    #prediction2 = elmnet(xtest).squeeze(-1)
    bpnet.load_state_dict(torch.load('ipsoBP_params.pkl'))      # 加载IPSO优化的BP神经网络参数
    prediction3 = bpnet(xtest).squeeze(-1)                      # IPSO-BP神经网络预测
    elmnet.load_state_dict(torch.load('ipsoELM_params.pkl'))    # 加载IPSO优化的ELM神经网络参数
    prediction4 = elmnet(xtest).squeeze(-1)                     # IPSO-ELM神经网络预测
    
    # 绘制预测结果对比图
    if flag == 'hh':
        plt.title('bohai_preswh')  # 黄海预测结果标题
    else:
        plt.title('yellow_preswh') # 渤海预测结果标题
    plt.xlabel('data_no')          # X轴表示数据点序号
    plt.ylabel('swh_pre')          # Y轴表示预测的海浪高度
    
    # 绘制真实值和各模型预测值曲线
    plt.plot(ytest.data.numpy(),'g',lw=1.5,label='truth')          # 绘制真实值曲线(绿色)
    plt.plot(prediction1.data.numpy(),'r',lw=1.5,label='bpnn')     # 绘制BP神经网络预测值曲线(红色)
    #plt.plot(prediction2.data.numpy(),'g-')
    plt.plot(prediction3.data.numpy(),'b',lw=1.5,label='ipso-bp')  # 绘制IPSO-BP预测值曲线(蓝色)
    plt.plot(prediction4.data.numpy(),'gray',lw=1.5,label='ipso-elm')  # 绘制IPSO-ELM预测值曲线(灰色)
    plt.legend(loc='upper left')  # 添加图例
    plt.show()

# 特定时间点海浪高度预测结果比较的注释代码
yhbtest = []
xhbtest = []
for i in range(30):
    yhbtest.append(labeltest[i*48]);
    xhbtest.append(x_test[i*48]);     #黄海8:00数据
swhfullin(xhbtest,yhbtest,'hh')
for i in range(30):
    yhbtest[i] = labeltest[i*48+6];
    xhbtest[i] = x_test[i*48+6];     #渤海8:00数据
swhfullin(xhbtest,yhbtest,'bh')