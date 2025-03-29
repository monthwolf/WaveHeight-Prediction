import torch 
#from torch.autograd import Variable 
import matplotlib.pyplot as plt 
import numpy as np
from swhplot import caloss
from data_cache import *

# 定义完整神经网络结构：2个输入节点，5个隐层节点，1个输出节点
net = torch.nn.Sequential( 
    torch.nn.Linear(2, 5),   # 输入层到隐藏层，输入特征为风速和气压
    torch.nn.Sigmoid(),      # 隐藏层激活函数，ELM算法通常使用sigmoid
    torch.nn.Linear(5, 1)    # 隐藏层到输出层，无激活函数
    #torch.nn.Softplus()
    )

# 定义仅包含输入层到隐藏层的网络结构，用于ELM算法的隐层计算
ihlayer = torch.nn.Sequential( 
    torch.nn.Linear(2, 5),   # 输入层到隐藏层，与完整网络保持一致
    torch.nn.Sigmoid()       # 隐藏层激活函数
    )

# 粒子群算法参数设置
swarmsize = 100      # 粒子数量
partlen = 15         # 每个粒子维度，对应隐藏层参数个数：(2x5)+(5)=15，不同于BP，ELM的输出层权重通过解析解获得
wmax,wmin = 0.9,0.4  # 惯性权重最大值和最小值
c1i = c2f = 2.5      # 学习因子初始值
c1f = c2i = 0.5      # 学习因子终值
Iter = 200           # 最大迭代次数

# 自适应惯性权重计算函数
def getwgh(fitness,i):
    """
    计算每个粒子的惯性权重，根据粒子当前适应值与种群平均适应值的关系动态调整
    参数:
        fitness: 所有粒子的适应值向量
        i: 当前粒子索引
    返回:
        w: 计算得到的惯性权重值
    """
    sum = 0
    for j in fitness:
        sum += j
    if fitness[i] <= sum/swarmsize:
        # 当粒子适应值低于平均适应值时，根据公式计算权重
        w = wmin + (wmax - wmin*(fitness[i] - fitness.min()))/(sum/swarmsize - fitness.min())
    else:
        # 当粒子适应值高于平均适应值时，使用最大权重
        w = wmax
    return w

# 动态学习因子计算函数
def getc1c2(iter):
    """
    根据当前迭代次数动态计算学习因子c1和c2
    c1控制粒子向个体最优解学习的程度
    c2控制粒子向全局最优解学习的程度
    参数:
        iter: 当前迭代次数
    返回:
        c1, c2: 当前迭代的学习因子值
    """
    c1 = (c1i - c1f)*(Iter - iter)/Iter + c1f
    c2 = (c2i - c2f)*(Iter - iter)/Iter + c2f
    return c1,c2

# 生成随机位置或速度的函数
def getrange():
    """
    生成随机的粒子位置或速度值
    返回:
        randompv: (-2,2)范围内的随机数
    """
    randompv = (np.random.rand()-0.5)*4    # 返回（-2，2）范围内的随机数
    return randompv

# 初始化粒子群位置和速度
def initswarm():
    """
    初始化粒子群的位置和速度矩阵
    返回:
        vswarm: 粒子速度矩阵，大小为swarmsize × partlen
        pswarm: 粒子位置矩阵，大小为swarmsize × partlen
    """
    vswarm,pswarm = np.zeros((swarmsize,partlen)),np.zeros((swarmsize,partlen))
    for i in range(swarmsize):
        for j in range(partlen):
            vswarm[i][j] = getrange()
            pswarm[i][j] = getrange()
    return vswarm,pswarm
    
# 计算每个粒子的适应度值（神经网络误差）
def getfitness(pswarm):
    """
    计算每个粒子的适应度值，这里使用神经网络的均方误差作为适应度
    ELM算法中，只优化输入权重和偏置，输出权重通过Moore-Penrose广义逆计算
    参数:
        pswarm: 粒子位置矩阵，表示输入层到隐藏层的神经网络参数
    返回:
        fitness: 所有粒子的适应度值向量
    """
    fitness = np.zeros(swarmsize)
    loss_function = torch.nn.MSELoss()
    for i in range(swarmsize):
        params = pswarm[i]
        
        # 将粒子位置参数应用到隐藏层网络上
        ihlayer.state_dict()['0.weight'].copy_(torch.tensor(np.array(params[0:10:1]).reshape(5,2)))   # 输入层到隐藏层权重
        ihlayer.state_dict()['0.bias'].copy_(torch.tensor(params[10:15:1]))                          # 隐藏层偏置
        
        # ELM算法的核心：计算隐含层输出并求解输出权重
        hiddenout = ihlayer(xtrain).squeeze(-1)    # 计算隐含层输出
        H = np.linalg.pinv(hiddenout.data.numpy().reshape(len(labeltrain),5))    # 求广义逆矩阵（Moore-Penrose伪逆）
        T = labeltrain.data.numpy().reshape(len(labeltrain))                     # 目标输出矩阵转置
        beta = np.dot(H,T)                                                       # 计算输出权重（β=H†T）
        beta = torch.tensor(beta).float()
        
        # 将输入权重、偏置和计算得到的输出权重应用到完整网络
        net.state_dict()['0.weight'].copy_(ihlayer.state_dict()['0.weight'])     # 复制输入层权重
        net.state_dict()['0.bias'].copy_(ihlayer.state_dict()['0.bias'])         # 复制隐藏层偏置
        net.state_dict()['2.weight'].copy_(beta)                                 # 设置计算得到的输出层权重
        net.state_dict()['2.bias'].copy_(torch.tensor(0))                        # 输出层偏置设为0
        
        # 计算预测结果和适应度
        prediction = net(xtrain) 
        prediction = prediction.squeeze(-1)
        fitness[i] = loss_function(prediction, labeltrain)                       # 计算MSE作为适应度
    return fitness

# 获取全局最优粒子及其适应度
def getpgfit(fitness,pswarm):
    """
    获取当前种群中适应度最小（最优）的粒子及其适应度值
    参数:
        fitness: 所有粒子的适应度值向量
        pswarm: 粒子位置矩阵
    返回:
        pg: 全局最优粒子的位置
        pgfitness: 全局最优粒子的适应度值
    """
    pgfitness = fitness.min()
    pg = pswarm[fitness.argmin()].copy()
    return pg,pgfitness

# 粒子群优化主函数
def optimi():
    """
    粒子群优化算法主函数
    使用PSO算法优化ELM神经网络的输入权重和偏置，寻找最优网络配置
    """
    # 初始化粒子群
    vswarm,pswarm = initswarm()
    fitness = getfitness(pswarm)
    pg,pgfit = getpgfit(fitness,pswarm)
    pi,pifit = pswarm.copy(),fitness.copy()    # pi和pifit分别存储每个粒子的历史最优位置和适应度
    pgfitlist = []      # 存放迭代过程中的全局最优粒子适应值
    
    # 迭代优化过程
    for iter in range(Iter):
        if pgfit <= 0.01:  # 当达到精度要求时提前结束
            break
        # 更新速度和位置
        for i in range(swarmsize):
            weight = getwgh(fitness,i)  # 计算自适应惯性权重
            c1,c2 = getc1c2(iter)       # 计算动态学习因子
            for j in range(partlen):
                # 更新粒子速度：惯性项 + 认知项 + 社会项
                vswarm[i][j] = weight*vswarm[i][j] + c1*np.random.rand()*(pi[i][j]-pswarm[i][j]) + c2*np.random.rand()*(pg[j]-pswarm[i][j])
                # 更新粒子位置
                pswarm[i][j] = pswarm[i][j] + vswarm[i][j]
        # 更新适应值
        fitness = getfitness(pswarm)
        # 更新全局最优粒子
        pg,pgfit = getpgfit(fitness,pswarm)
        pgfitlist.append(pgfit)
        # 更新局部最优粒子
        for i in range(swarmsize):
            if fitness[i] < pifit[i]:
                pifit[i] = fitness[i].copy()
                pi[i] = pswarm[i].copy()
    
    # 绘制粒子搜索过程全局最优的适应值变化
    plt.title('swarm_fit')  
    plt.plot(pgfitlist) 
    plt.ylabel('pg_fitness')
    plt.xlabel('iter_num')
    plt.show()
    
    # 迭代结束后，再次检查是否存在更好的局部最优解
    for j in range(swarmsize):   
        if pifit[j] < pgfit:
            pgfit = pifit[j].copy()
            pg = pi[j].copy()
            
    # 优化完成，使用最优输入权重和偏置
    ihlayer.state_dict()['0.weight'].copy_(torch.tensor(np.array(pg[0:10:1]).reshape(5,2)))
    ihlayer.state_dict()['0.bias'].copy_(torch.tensor(pg[10:15:1]))
    
    # 计算最终隐含层输出并求解输出权重
    hiddenout = ihlayer(xtrain).squeeze(-1)    # 计算隐含层输出
    H = np.linalg.pinv(hiddenout.data.numpy().reshape(len(labeltrain),5))    # 求广义逆
    T = labeltrain.data.numpy().reshape(len(labeltrain))                     # 矩阵转置
    beta = np.dot(H,T)                                                       # 矩阵相乘计算输出权重
    beta = torch.tensor(beta).float()
    
    # 将优化后的参数应用到完整网络
    net.state_dict()['0.weight'].copy_(ihlayer.state_dict()['0.weight'])
    net.state_dict()['0.bias'].copy_(ihlayer.state_dict()['0.bias'])
    net.state_dict()['2.weight'].copy_(beta)
    net.state_dict()['2.bias'].copy_(torch.tensor(0))
    
    # 保存优化后的网络参数
    torch.save(net.state_dict(), 'ipsoELM_params.pkl')

# 加载模型参数并测试模型性能
def reload_params(): 
    """
    加载训练好的模型参数，在测试集和训练集上评估模型性能
    """
    # 载入神经网络的模型参数
    net.load_state_dict(torch.load('ipsoELM_params.pkl')) 
    loss_function = torch.nn.MSELoss()
    
    # 在测试集上进行预测
    prediction = net(xtest)
    prediction = prediction.squeeze(-1)
    
    # 计算测试集上的各项评价指标
    caloss(prediction,labeltest)
    
    # 绘制测试集预测结果散点图与拟合线
    plt.title('test_net') 
    param = np.polyfit(labeltest.squeeze(-1).data.numpy(), prediction.squeeze(-1).data.numpy(),1)
    p = np.poly1d(param,variable='x')
    rsquare = 1 - loss_function(labeltest,prediction).data.numpy()/np.var(labeltest.data.numpy())    # 计算R方（决定系数）
    plt.scatter(labeltest.data.numpy(), prediction.data.numpy()) 
    plt.xlabel('ytest_label')
    plt.ylabel('ytest_prediction')
    plt.plot(labeltest.data.numpy(), p(labeltest.data.numpy()),'r--') 
    plt.text(max(labeltest.data),max(prediction.data),'y='+str(p).strip()+'\nRsquare='+str(round(rsquare,4)),verticalalignment="top",horizontalalignment="right")
    plt.show()

    # 在训练集上进行预测
    prediction = net(xtrain)
    prediction = prediction.squeeze(-1)
    
    # 计算训练集上的各项评价指标
    caloss(prediction,labeltrain)
    
    # 绘制训练集预测结果散点图与拟合线
    plt.title('train_net') 
    param = np.polyfit(labeltrain.squeeze(-1).data.numpy(), prediction.squeeze(-1).data.numpy(),1)
    p = np.poly1d(param,variable='x')
    rsquare = 1 - loss_function(labeltrain,prediction).data.numpy()/np.var(labeltrain.data.numpy())    # 计算R方
    plt.scatter(labeltrain.data.numpy(), prediction.data.numpy()) 
    plt.xlabel('ytrain_label')
    plt.ylabel('ytrain_prediction')
    plt.plot(labeltrain.data.numpy(), p(labeltrain.data.numpy()),'r--') 
    plt.text(max(labeltrain.data),max(prediction.data),'y='+str(p).strip()+'\nRsquare='+str(round(rsquare,4)),verticalalignment="top",horizontalalignment="right")
    plt.show()

# 程序执行入口：先优化网络参数，然后加载并测试
optimi()
reload_params()