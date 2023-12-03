import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np


class IrisNet(nn.Module):
    def __init__(self):
        super(IrisNet, self).__init__()
        self.fc1 = nn.Linear(4, 8)
        self.act1 = nn.LeakyReLU()
        self.fc2 = nn.Linear(8, 8)
        self.act2 = nn.LeakyReLU()
        self.fc3 = nn.Linear(8, 8)
        self.act3 = nn.LeakyReLU()
        self.fc4 = nn.Linear(8, 3)

    def forward(self, x):
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        x = self.act3(self.fc3(x))
        x = self.fc4(x)
        return x


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)
        self.act1 = nn.Sigmoid()
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.act2 = nn.Sigmoid()
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.act3 = nn.Sigmoid()
        self.fc2 = nn.Linear(120, 84)
        self.act4 = nn.Sigmoid()
        self.fc3 = nn.Linear(84, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.act1(self.conv1(x))
        x = self.pool1(x)
        x = self.act2(self.conv2(x))
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.act3(self.fc1(x))
        x = self.act4(self.fc2(x))
        x = self.fc3(x)
        x = self.softmax(x)
        return x


def iris_test():
    iris = load_iris()
    x = iris.data
    y = iris.target

    scalar = StandardScaler()
    x = scalar.fit_transform(x)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=20, random_state=3)

    x_train_tensor = torch.FloatTensor(x_train)
    y_train_tensor = torch.LongTensor(y_train)
    x_test_tensor = torch.FloatTensor(x_test)
    y_test_tensor = torch.LongTensor(y_test)

    input_size = 4
    hidden_size = 32
    output_size = 3
    learning_rate = 1
    epochs = 200

    model = IrisNet()
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    train_losses = []
    train_accs = []
    test_accs = []
    # print('{0:>8}  {1:>8}  {2:>9}  {3:>9}'.format('Epoch', 'Loss', 'Train Acc', 'Test Acc'))
    for epoch in range(epochs):
        model.train()

        optimizer.zero_grad()
        outputs = model(x_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            train_outputs = model(x_train_tensor)
            _, train_pred = torch.max(train_outputs, 1)
            train_accuracy = accuracy_score(y_train, train_pred.numpy())
            train_accs.append(train_accuracy)

            test_outputs = model(x_test_tensor)
            _, test_pred = torch.max(test_outputs, 1)
            test_accuracy = accuracy_score(y_test, test_pred.numpy())
            test_accs.append(test_accuracy)

    # print(f'Epoch: {epochs:>8}, Loss: {train_losses[-1]:>8.4f}, Train Acc: {train_accs[-1]:9.4f}, Test Acc: {test_accs[-1]:9.4f}')
    print(
        f'Epoch: {epochs}, Loss: {train_losses[-1]:.4f}, Train Acc: {train_accs[-1]:.4f}, Test Acc: {test_accs[-1]:.4f}')

    fig = plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(test_accs, label='Test Accuracy')
    plt.title('Training and Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.suptitle(f'4-8(LR)-8(LR)-8(LR)-3, lr={learning_rate}(SGD)')
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.25, hspace=0)

    plt.show()


def mnist_test():
    # 将图片标准化
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    # 检查GPU是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化模型、损失函数和优化器
    model = LeNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    epochs = 10
    train_losses = []
    train_accs = []
    test_accs = []

    print('{0:>8}  {1:>8}  {2:>9}  {3:>9}'.format('Epoch', 'Loss', 'Train Acc', 'Test Acc'))
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        # 1个batch
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = correct_train / total_train
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # 在测试集上计算精度
        model.eval()
        correct_test = 0
        total_test = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()

        test_acc = correct_test / total_test
        test_accs.append(test_acc)

        print(f'{epoch + 1:>5}/{epochs}  {train_loss:>8.4f}  {train_acc:9.4f}  {test_acc:9.4f}')

    # 画出训练曲线
    fig = plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(test_accs, label='Test Accuracy')
    plt.title('Training and Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    fig.tight_layout()
    plt.subplots_adjust(wspace=0.25, hspace=0)

    plt.show()

    # 在测试集上随机抽取10个样本查看分类结果
    model.eval()
    print('{0:>10}  {1:>12}  {2:>10}'.format('Test Case', 'Ground Truth', 'Predicted'))
    with torch.no_grad():
        for i in range(10):
            idx = np.random.randint(0, len(test_dataset))
            sample, label = test_dataset[idx]
            sample, label = sample.to(device), int(label)
            sample = sample.unsqueeze(0)
            output = model(sample)
            _, predicted = torch.max(output.data, 1)
            print(f'{idx:>10}  {label:>12}  {predicted.item():>10}')


# iris_test()
mnist_test()
