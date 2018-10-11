import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from torch.autograd import Variable


def restore_net():
    x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
                        [9.779], [6.182], [7.59], [2.167], [7.042],
                        [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)
    y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
                        [3.336], [2.596], [2.53], [1.221], [2.827],
                        [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)
    x_train = torch.from_numpy(x_train)
    y_train = torch.from_numpy(y_train)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net2 = torch.load('./model/test.pth').to(device)
    predict = net2(Variable(x_train).cuda())
    predict = predict.data.cpu().numpy()
    plt.plot(x_train.numpy(), y_train.numpy(), 'ro', label='Original data')
    plt.plot(x_train.numpy(), predict, label='Fitting Line')
    plt.show()


def restore_params():
    x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
                        [9.779], [6.182], [7.59], [2.167], [7.042],
                        [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)
    y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
                        [3.336], [2.596], [2.53], [1.221], [2.827],
                        [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)
    x_train = torch.from_numpy(x_train)
    y_train = torch.from_numpy(y_train)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net3 = LinearRegression().to(device)
    net3.load_state_dict(torch.load('./model/test_param.pth'))
    predict = net3(Variable(x_train).cuda())
    predict = predict.data.cpu().numpy()
    plt.plot(x_train.numpy(), y_train.numpy(), 'ro', label='Original data')
    plt.plot(x_train.numpy(), predict, label='Fitting Line')
    plt.show()


class LinearRegression(nn.Module):
    def __init__(self):
        # super()函数，调用父类的一个方法。首先找到父类nn.Module，然后把类对象LinearRegression转换为类nn.Module的对象
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        out = self.linear(x)
        return out


restore_net()
restore_params()
