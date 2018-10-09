'''
逻辑回归
注：此代码时常陷入梯度不变，暂时未搞清原理，可能是权重随机初始值不合适
'''
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from torch import optim
from torch.autograd import Variable

with open('./data/data.txt', 'r') as f:
    data_list = f.readlines()
    # 读取一行数据，并将每行数据以string形式存入列表
    # 示例：['34.62,78.02,0\n', '30.28,43.89,0\n', ...]
    data_list = [i.split('\n')[0] for i in data_list]
    # 删除掉每个列表元素中的'\n'
    # 首先将'34.62,78.02,0\n'切分成['34.62,78.02,0', '']，再将非空的第0位保存
    # 示例：['34.62,78.02,0', '30.28,43.89,0', ...]
    data_list = [i.split(',') for i in data_list]
    # 以','为分隔符，切分字符串，返回分割后的字符串列表
    # 示例：[['34.62','78.02','0'], ['30.28','43.89','0'], ...]
    data = [(float(i[0]), float(i[1]), float(i[2])) for i in data_list]
    # 合并数据集，并将string转换为float型
    # 示例：[(34.62,78.02,0), (30.28,43.89,0), ...]


x0 = list(filter(lambda x: x[-1]== 0.0, data))  # 过滤掉每个数据不为0的数据
'''
filter(function, iterable)，用于过滤序列，参数1为一个函数，参数2为序列
把序列中的每个元素赋值到参数1中
# 例子：
# def is_odd(n):
#     return n % 2 == 1
# newlist = filter(is_odd, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# print(newlist)
# 输出结果：[1, 3, 5, 7, 9]
'''
'''
func=lambda x:x+1
等价于
# def func(x):
#     return x+1
'''
x1 = list(filter(lambda x: x[-1]== 1.0, data))  # 过滤掉每个数据不为1的数据
plot_x0_0 = [i[0] for i in x0]
plot_x0_1 = [i[1] for i in x0]
plot_x1_0 = [i[0] for i in x1]
plot_x1_1 = [i[1] for i in x1]
plt.plot(plot_x0_0, plot_x0_1, 'ro', label= 'x0')
plt.plot(plot_x1_0, plot_x1_1, 'bo', label= 'x1')
plt.legend(loc='best')  # 显示图例，loc参数设置图例显示位置，'best'关键字为自适应方式
plt.show()
# 构造训练集
x_data = []
y_data = []
for num in data:
    x_data.append([num[0],num[1]])
    y_data.append([num[2]])
x_data = torch.from_numpy(np.array(x_data, dtype=np.float32))
y_data = torch.from_numpy(np.array(y_data, dtype=np.float32))


class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.lr = nn.Linear(2, 1)
        self.sm = nn.Sigmoid()

    def forward(self, x):
        x = self.lr(x)
        x = self.sm(x)
        return x


device = torch.device("cuda")
logistic_model = LogisticRegression().to(device)
criterion = nn.BCELoss()
optimizer = optim.SGD(logistic_model.parameters(), lr=1e-3, momentum=0.9)
for epoch in range(50000):
    x = Variable(x_data).cuda()
    y = Variable(y_data).cuda()
    #forward
    out = logistic_model(x)
    loss = criterion(out, y)
    print_loss = loss.item()
    # backward
    optimizer.zero_grad()  # 初始化梯度
    loss.backward()  # 反向传播
    optimizer.step()  # 反馈计算梯度并更新权值

    if (epoch+1)%1000 == 0:
        print('*'*10)
        print('Epoch[{}], loss: {:.4f}'.format(epoch+1, print_loss))
        ##数据可视化
        # params = list(logistic_model.parameters())
        # w0 = float(params[0].data.cpu().numpy()[0][0])
        # w1 = float(params[0].data.cpu().numpy()[0][1])
        # b = float(params[1].data.cpu().numpy())
        # plt.cla()
        # plot_x = np.arange(30, 100, 0.1)
        # plot_y = (-w0 * plot_x - b) / w1
        # plot_x0_0 = [i[0] for i in x0]
        # plot_x0_1 = [i[1] for i in x0]
        # plot_x1_0 = [i[0] for i in x1]
        # plot_x1_1 = [i[1] for i in x1]
        # plt.plot(plot_x0_0, plot_x0_1, 'ro', label='x0')
        # plt.plot(plot_x1_0, plot_x1_1, 'bo', label='x1')
        # plt.plot(plot_x, plot_y)
        # plt.xlim(0, 150)
        # plt.ylim(0, 150)
        # plt.pause(0.1)
