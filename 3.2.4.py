import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from torch import optim
from torch.autograd import Variable


# 生成数据集
x_train = np.array([[3.3], [4.4],[5.5],[6.71],[6.93],[4.168],
                    [9.779],[6.182],[7.59],[2.167],[7.042],
                    [10.791],[5.313],[7.997],[3.1]],dtype=np.float32)
y_train = np.array([[1.7], [2.76],[2.09],[3.19],[1.694],[1.573],
                    [3.336],[2.596],[2.53],[1.221],[2.827],
                    [3.465],[1.65],[2.904],[1.3]],dtype=np.float32)
# 训练集可视化
plt.plot(x_train, y_train, 'ro', label='train')
plt.show()
# numpy转torch
x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train)


class LinearRegression(nn.Module):
    def __init__(self):
        # super()函数，调用父类的一个方法。首先找到父类nn.Module，然后把类对象LinearRegression转换为类nn.Module的对象
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        out = self.linear(x)
        return out


# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 定义模型
model = LinearRegression().to(device)
# 定义损失函数，均方误差
criterion = nn.MSELoss()
params = list(model.parameters())
print(params)  # 输出待学习参数，params[0]为w，params[1]为b
# 定义优化函数，梯度下降
optimizer = optim.SGD(model.parameters(), lr=1e-3)  # 1e-3=10^(-3)=0.001
# model.parameters(),存储要优化的参数，w与b
# lr: Learning rate
# w' = w - lr * d(Loss)/dw
# b' = b - lr * d(Loss)/db

num_epochs = 2000
for epoch in range(num_epochs):
    inputs = Variable(x_train).cuda()
    target = Variable(y_train).cuda()
    # forward
    out = model(inputs)  # 得到前向传播结果，由LinearRegression.forward()函数返回
    loss = criterion(out, target)  # 得到损失函数
    # 均方差Loss=∑[f(xi)-yi]^2，out=f(xi),yi=target

    # backward
    optimizer.zero_grad()  # 初始化梯度
    loss.backward()  # 反向传播
    optimizer.step()  # 反馈计算梯度并更新权值
    if (epoch+1)%20==0:
        print('Epoch[{}/{}], loss: {:.6f}'.format(epoch+1, num_epochs, loss.data[0]))
        # loss是Variable数据类型，可以通过loss.data取出一个Tensor数据，再通过loss.data[0]打印出int或float型数据

params = list(model.parameters())
print(params)  # 输出学习好的参数
model.eval()
# 初始化使用gpu，因此必须将变量变为Variable(x_train).cuda()，不能直接使用Variable(x_train)
predict = model(Variable(x_train).cuda())
# predict = predict.data.numpy()会报错，需要先将cuda()转换为cpu()格式才能转换为numpy
predict = predict.data.cpu().numpy()
plt.plot(x_train.numpy(), y_train.numpy(), 'ro', label='Original data')
plt.plot(x_train.numpy(), predict, label='Fitting Line')
plt.show()
