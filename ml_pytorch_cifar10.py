"""
ffzs
2018.1.16
win10
i7-6700HQ
GTX965M
"""
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import transforms, datasets
import visdom
import time
import numpy as np
from models import resnet18

viz = visdom.Visdom()

BATCH_SIZE = 20
EPOCHS = 10
LR = 0.005

USE_GPU = True

if USE_GPU:
    gpu_status = torch.cuda.is_available()
else:
    gpu_status = False
transform_train = transforms.Compose(
     # 图像翻转
    [transforms.RandomHorizontalFlip(),
     # 数据张量化 (0,255) >> (0,1)
     transforms.ToTensor(),
     # 数据归一处理 正态分布 (0,1） >> (-1,1)
     transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
transform_test = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_dataset = datasets.CIFAR10('./data', train=True, transform=transform_train)
test_dataset = datasets.CIFAR10('./data', train=False, transform=transform_test)

train_loader = DataLoader(train_dataset, BATCH_SIZE, True)
test_loader = DataLoader(test_dataset, BATCH_SIZE, False)

classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

dataiter = iter(test_loader)
image, label = dataiter.next()
# 可视化visualize
image = viz.images(image[:10]/2+0.5, nrow=10, padding=3, env='cifar10')
text = viz.text('||'.join('%6s' % classes[label[j]] for j in range(10)),env='cifar10')

class CNN(nn.Module):
    def __init__(self, in_dim, n_class):
        super(CNN, self).__init__()
        # 卷积部分
        self.cnn = nn.Sequential(
            nn.BatchNorm2d(in_dim),
            nn.ReLU(True),
            nn.Conv2d(in_dim, 16, 5, 1, 2),  # (32,32)
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # (32,32) >> (16,16)
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # (16,16) >> (8,8)
        )
        # linear 部分
        self.fc = nn.Sequential(
            nn.BatchNorm2d(32*8*8),
            nn.ReLU(True),
            nn.Linear(32*8*8, 120),
            nn.BatchNorm2d(120),
            nn.ReLU(True),
            nn.Linear(120, 50),
            nn.BatchNorm2d(50),
            nn.ReLU(True),
            nn.Linear(50, n_class),
        )
    def forward(self, x):
        out = self.cnn(x)
        out = self.fc(out.view(-1, 32*8*8)) # 通过 view改变out形态
        return out

# net = CNN(3, 10)

net = resnet18(True)
num_ftrs = net.fc.in_features
net.fc = nn.Linear(num_ftrs, 10)

print(net)

if gpu_status:
    net = net.cuda()
    print("使用gpu")
else:
    print("使用cpu")
# 交叉熵
loss_f = nn.CrossEntropyLoss()
# SGD加动量加快优化
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9)
# learning-rate 变化函数
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# 隔多少个batch输出数据
tr_num = len(train_dataset)/BATCH_SIZE/5
ts_num = len(test_dataset)/BATCH_SIZE/5
# 开始时间
start_time = time.time()
# visdom 创建 line 窗口
line = viz.line(Y=np.arange(10), env="cifar10")
# 记录数据的一些状态
tr_loss, ts_loss, tr_acc, ts_acc, step = [], [], [], [], []
# 记录net最佳状态
best_acc = 0.
best_state = net.state_dict()
for epoch in range(EPOCHS):
    # 检测运行中loss ，acc 变化
    running_loss, running_acc = 0.0, 0.
    scheduler.step()
    # 训练环境设定
    net.train()
    for i, (img, label) in enumerate(train_loader, 1):
        if gpu_status:
            img, label = img.cuda(), label.cuda()
        img, label = Variable(img), Variable(label)
        out = net(img)
        loss = loss_f(out, label)
        pred = torch.max(out, 1)[1]
        running_acc += sum(pred==label).data[0]
        running_loss += loss.data[0]*len(label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 数据输出，观察运行状态
        if i % tr_num == 0:
            print("TRAIN : [{}/{}] | loss: {:.4f} | r_acc: {:.4f} ".format(epoch+1, EPOCHS, running_loss/(tr_num*BATCH_SIZE),
                                                                running_acc/(tr_num*BATCH_SIZE)))
            # 训练loss，acc 可视化数据收集
            tr_loss.append(running_loss/(tr_num*BATCH_SIZE))
            tr_acc.append(running_acc/(tr_num*BATCH_SIZE))
            running_loss, running_acc = 0.0, 0.

    net.eval()
    eval_loss, eval_acc = 0., 0.
    for i, (img, label) in enumerate(test_loader, 1):
        if gpu_status:
            img, label = img.cuda(), label.cuda()
        # 因为是测试集 volatile1 不影响训练状态
        img, label = Variable(img, volatile=True), Variable(label, volatile=True)
        out = net(img)
        loss = loss_f(out, label)
        pred = torch.max(out, 1)[1]
        eval_acc += sum(pred == label).data[0]
        eval_loss += loss.data[0]*len(label)
        if i % ts_num == 0:
            print("test : [{}/{}] | loss: {:.4f} | r_acc: {:.4f} ".format(epoch + 1, EPOCHS,
                                                                            eval_loss / (ts_num*BATCH_SIZE),
                                                                            eval_acc / (ts_num*BATCH_SIZE)))
            ts_loss.append(eval_loss / (ts_num*BATCH_SIZE))
            ts_acc.append(eval_acc / (ts_num*BATCH_SIZE))
            eval_loss, eval_acc = 0., 0.
    # visualize
    viz.line(Y=np.column_stack((np.array(tr_loss), np.array(tr_acc), np.array(ts_loss), np.array(ts_acc))),
             win=line,
             opts=dict(legend=["tr_loss", "tr_acc", "ts_loss", "ts_acc"],
                       title="cafar10"),
             env="cifar10")
    # 保存训练最佳状态
    if eval_acc / (len(test_dataset)) > best_acc:
        best_acc = eval_acc / (len(test_dataset))
        best_state = net.state_dict()

net.eval()
eval_loss, eval_acc = 0., 0.
match = [0]*100
# 因为要记录每一个图片的测试结果，batch_size设为1
test_loader = DataLoader(test_dataset, 1, False)
for i, (img, label) in enumerate(test_loader, 1):
    if gpu_status:
        img, label = img.cuda(), label.cuda()
    img, label = Variable(img, volatile=True), Variable(label, volatile=True)
    out = net(img)
    loss = loss_f(out, label)
    pred = torch.max(out, 1)[1]
    eval_acc += sum(pred == label).data[0]
    eval_loss += loss.data[0]
    # 对 测试集 数据准确率进行记录 注意：本数据集为等比例 每个分类数量相等，否者要用占比
    number = int(label.data.cpu().numpy()[0]*10+pred.data.cpu().numpy()[0])
    match[number] = match[number] + 1
    if i % 1000 == 0:
        print("{} | loss : {:.4f} | acc : {:.4f}|time:{:.1f}".format(i, eval_loss/i, eval_acc/i,time.time()-start_time))
count = np.array(match).reshape(10,10)
viz.heatmap(X=count, opts=dict(
        columnnames=classes, # 添加分类
        rownames=classes,
        colormap='Jet', # 选取colormap 用颜色梯度 可视 数值梯度
        title="ACC: {:.4f}".format(eval_acc/len(test_dataset)), #标题
        xlabel="pred",
        ylabel="label"),
        env="cifar10")

if eval_acc/len(test_dataset) > 0.75:
    torch.save(best_state, 'save/cifar10.pth')




