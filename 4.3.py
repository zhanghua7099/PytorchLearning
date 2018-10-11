import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Sequential(         # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=1,              # input height
                out_channels=16,            # n_filters
                kernel_size=5,              # filter size
                stride=1,                   # filter movement/step
                padding=2,                  # if want same width and length of this image after con2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (16, 28, 28)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (16, 14, 14)
        )
        self.conv2 = nn.Sequential(         # input shape (16, 14, 14)
            nn.Conv2d(16, 32, 5, 1, 2),     # output shape (32, 14, 14)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(2),                # output shape (32, 7, 7)
        )
        self.out = nn.Linear(32 * 7 * 7, 10)   # fully connected layer, output 10 classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)           # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        output = self.out(x)
        return output


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_epochs = 5
batch_size = 100
learning_rate = 0.001
train_dataset = torchvision.datasets.MNIST(root='./data/mnist',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=False)

test_dataset = torchvision.datasets.MNIST(root='./data/mnist',
                                          train=False,
                                          transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Move tensors to the configured device
        images = images.to(device)
        labels = labels.to(device)

        # 前向传播和计算loss
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 后向传播和调整参数
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 每100个batch打印一次数据
        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

# 模型测试部分
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        # labels是Tensor数据类型
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        # print(labels)
        # tensor([9, 6, 2, 3, 7, 1, 9, 2, 2, 5, 3, 7, 8, 0, 1, 2, 3, 4, 7, 8, 9, 0, 1, 2,
        #         3, 4, 7, 8, 9, 0, 1, 7, 8, 9, 8, 9, 2, 6, 1, 3, 5, 4, 8, 2, 6, 4, 3, 4,
        #         5, 9, 2, 0, 3, 9, 4, 9, 7, 3, 8, 7, 4, 4, 9, 8, 5, 8, 2, 6, 6, 2, 3, 1,
        #         3, 2, 7, 3, 1, 9, 0, 1, 1, 3, 5, 0, 7, 8, 1, 5, 1, 4, 6, 0, 0, 4, 9, 1,
        #         6, 6, 9, 0], device='cuda:0')

        _, predicted = torch.max(outputs.data, 1)  # 返回输入张量给定维度上每行的最大值，并同时返回每个最大值的位置索引。
        # print(predicted)
        # tensor([8, 9, 0, 1, 8, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 7, 8, 9, 7, 8, 6, 4, 1,
        #         9, 3, 1, 4, 4, 7, 0, 1, 9, 2, 8, 7, 8, 2, 6, 0, 6, 5, 3, 3, 3, 9, 1, 4,
        #         0, 6, 1, 0, 0, 6, 2, 1, 1, 7, 7, 8, 4, 6, 0, 7, 0, 3, 6, 8, 7, 1, 5, 2,
        #         4, 9, 4, 3, 6, 4, 1, 7, 2, 6, 6, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2,
        #         3, 4, 5, 6], device='cuda:0')
        total += labels.size(0)  # 统计总数
        correct += (predicted == labels).sum().item()  # 正确则加1
        # test = (predicted == labels)
        # print(test)
        # 注：该判断方式为Tensor数据类型特有
        # tensor([1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        #         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        #         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        #         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        #         1, 1, 1, 1], device='cuda:0', dtype=torch.uint8)

    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))
