import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
 
 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 
# 定义超参数
input_size = 784
hidden_size = 500
num_classes = 10
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
 
# 添加批标准化
class Batch_Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Batch_Net, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(input_size, hidden_size), nn.BatchNorm1d(hidden_size), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(hidden_size, 20), nn.BatchNorm1d(20), nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(20, num_classes))

    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

 
model = Batch_Net(input_size, hidden_size, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  
 
# 训练模型
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):  
        # Move tensors to the configured device
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        
        # 前向传播和计算loss
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 后向传播和调整参数
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 每100个batch打印一次数据
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
 
# 模型测试部分
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
 
    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))
 
# # 保存模型参数
# torch.save(model.state_dict(), './model/mnist.ckpt')
