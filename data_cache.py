import torch
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler 
from ncread import *

# 创建数据 
labeltrain = torch.unsqueeze(torch.linspace(-1, 1, 7320), dim=1)
labeltest = torch.unsqueeze(torch.linspace(-1, 1, 1464), dim=1)
xtrain = []
xtest = []
mtrain = msl[0:7320:1]
mtest = msl[7320:8784:1]
uvtrain = uv10[0:7320:1]
uvtest = uv10[7320:8784:1]

for i in range(7320):
  labeltrain[i] = swh[i]
  xtrain.append([mtrain[i],uvtrain[i]])

for i in range(1464):
  labeltest[i] = swh[i+7320]
  xtest.append([mtest[i],uvtest[i]])
x_train = MinMaxScaler().fit_transform(xtrain)
x_test = MinMaxScaler().fit_transform(xtest)
xtrain = torch.unsqueeze(torch.tensor(x_train).float(), dim=1)   #升维
xtest = torch.unsqueeze(torch.tensor(x_test).float(), dim=1) 
#ytrain = y_train + 0.2*torch.rand(y_train.size())     #加入噪声提高鲁棒性
#ytest = y_test + 0.2*torch.rand(y_test.size()) 
xtrain, xtest = Variable(xtrain, requires_grad=True), Variable(xtest, requires_grad=True) 