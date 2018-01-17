"""
ffzs
2018.1.14
win10
i7-6700HQ
GTX965M
"""

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import transforms, datasets, models
import visdom
import time
import numpy as np


viz = visdom.Visdom()

BATCH_SIZE = 64
LR = 0.001
EPOCHS = 20

USE_GPU = True
if USE_GPU:
    gpu_status = torch.cuda.is_available()
else:
    gpu_status = False

train_dataset = datasets.MNIST('./mnist', True, transforms.ToTensor(), download=False)
test_dataset = datasets.MNIST('./mnist', False, transforms.ToTensor())
train_loader = DataLoader(train_dataset, BATCH_SIZE, True)
test_loader = DataLoader(test_dataset, 400, False)

dataiter = iter(train_loader)
inputs, labels = dataiter.next()
# 可视化visualize
viz.images(inputs[:16], nrow=8, padding=3)
time.sleep(0.5)
image = viz.images(inputs[:16], nrow=8, padding=3)

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.en_conv = nn.Sequential(
            nn.Conv2d(1, 16, 4, 2, 1),
            nn.BatchNorm2d(16),
            nn.Tanh(),
            nn.Conv2d(16, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.Tanh(),
            nn.Conv2d(32, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.Tanh()
        )
        self.en_fc = nn.Linear(16*7*7, 3)
        self.de_fc = nn.Linear(3, 16*7*7)
        self.de_conv = nn.Sequential(
            nn.ConvTranspose2d(16, 16, 4, 2, 1),
            nn.BatchNorm2d(16),
            nn.Tanh(),
            nn.ConvTranspose2d(16, 1, 4, 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        en = self.en_conv(x)
        code = self.en_fc(en.view(en.size(0), -1))
        de = self.de_fc(code)
        decoded = self.de_conv(de.view(de.size(0), 16, 7, 7))
        return code, decoded

net = AutoEncoder()

data = torch.Tensor(BATCH_SIZE ,28*28)
data = Variable(data)
if torch.cuda.is_available():
    net = net.cuda()
    data = data.cuda()

optimizer = torch.optim.Adam(net.parameters(), lr=LR)
loss_f = nn.MSELoss()

scatter=viz.scatter(X=np.random.rand(2, 2), Y=(np.random.rand(2) + 1.5).astype(int), opts=dict(showlegend=True))

for epoch in range(EPOCHS):
    net.train()
    for step, (images, _) in enumerate(train_loader, 1):
        net.zero_grad()
        data.data.resize_(images.size()).copy_(images)
        # data = data.view(-1, 28*28)
        code, decoded = net(data)
        loss = loss_f(decoded, data)
        loss.backward()
        optimizer.step()

        if step % 200 == 0:
            net.eval()
            eps = Variable(inputs)   #.view(-1, 28*28))
            if torch.cuda.is_available():
                eps = eps.cuda()
            tags, fake = net(eps)

            viz.images(fake[:16].data.cpu().view(-1, 1, 28, 28), win=image, nrow=8)
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, step * len(data), len(train_loader.dataset),
                       100. * step / len(train_loader),
                       loss.data[0]))
            # if step == 200:
               # viz.images(fake[:16].data.cpu().view(-1, 1, 28, 28), nrow=8 ,opts=dict(title="epoch:{}".format(epoch)))
               # viz.scatter(X=tags.data.cpu(), Y=labels + 1, win=scatter, opts=dict(showlegend=True))

for step, (images, labels) in enumerate(test_loader, 1):
    if step > 1:
        break
    if torch.cuda.is_available():
        images = images.cuda()
    images = Variable(images)
    tags, fake = net(images)
    viz.scatter(X=tags.data.cpu(), Y=labels + 1, win=scatter, opts=dict(showlegend=True))