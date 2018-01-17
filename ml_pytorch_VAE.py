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
from torchvision import transforms, datasets
import visdom
import time
import numpy as np

viz = visdom.Visdom()

BATCH_SIZE = 64
LR = 0.001
EPOCHS = 30
hidden_size = 3

USE_GPU = True
if USE_GPU:
    gpu_status = torch.cuda.is_available()
else:
    gpu_status = False

train_dataset = datasets.MNIST('./mnist', True, transforms.ToTensor(), download=False)
test_dataset = datasets.MNIST('./mnist', False, transforms.ToTensor())
train_loader = DataLoader(train_dataset, BATCH_SIZE, True)
test_loader = DataLoader(test_dataset, 400, False)

dataiter = iter(test_loader)
inputs, labels = dataiter.next()
# 可视化visualize
viz.images(inputs[:16], nrow=8, padding=3)
time.sleep(0.5)
image = viz.images(inputs[:16], nrow=8, padding=3)
time.sleep(0.5)
scatter=viz.scatter(X=np.random.rand(2, 2), Y=(np.random.rand(2) + 1.5).astype(int))

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(1, 16, 4, 2, 1),
                                   nn.BatchNorm2d(16),
                                   nn.ReLU(True))
        self.conv2 = nn.Sequential(nn.Conv2d(16, 32, 4, 2, 1),
                                   nn.BatchNorm2d(32),
                                   nn.ReLU(True))
        self.conv3 = nn.Sequential(nn.Conv2d(32, 16, 3, 1, 1),
                                   nn.BatchNorm2d(16),
                                   nn.ReLU(True))

        self.fc_encode1 = nn.Linear(16 * 7 * 7, hidden_size)
        self.fc_encode2 = nn.Linear(16 * 7 * 7, hidden_size)
        self.fc_decode = nn.Linear(hidden_size, 16 * 7 * 7)

        self.deconv1 = nn.Sequential(nn.ConvTranspose2d(16, 16, 4, 2, 1),
                                     nn.BatchNorm2d(16),
                                     nn.ReLU())
        self.deconv2 = nn.Sequential(nn.ConvTranspose2d(16, 1, 4, 2, 1),
                                     nn.Sigmoid())

    def encoder(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        # print(out)
        return self.fc_encode1(out.view(out.size(0), -1)), self.fc_encode2(out.view(out.size(0), -1))

    def sampler(self, mean, std):
        var = std.mul(0.5).exp_()
        eps = torch.FloatTensor(var.size()).normal_()
        eps = Variable(eps)
        if gpu_status:
            eps = eps.cuda()
        return eps.mul(var).add_(mean)

    def decoder(self, x):
        out = self.fc_decode(x)
        out = self.deconv1(out.view(x.size(0), 16, 7, 7))
        out = self.deconv2(out)
        return out

    def forward(self, x):
        mean, std = self.encoder(x)
        code = self.sampler(mean, std)
        out = self.decoder(code)
        return out, code, mean, std

net = VAE()

bce = nn.BCELoss()
bce.size_average = False
data = torch.Tensor(BATCH_SIZE ,28*28)
data = Variable(data)
if torch.cuda.is_available():
    net = net.cuda()
    bce = bce.cuda()
    data = data.cuda()

def loss_f(out, target, mean, std):
    bceloss = bce(out, target)
    latent_loss= torch.sum(mean.pow(2).add_(std.exp()).mul_(-1).add_(1).add_(std)).mul_(-0.5)
    return bceloss + latent_loss

optimizer = torch.optim.Adam(net.parameters(), lr=LR)

for epoch in range(EPOCHS):
    net.train()
    for step, (images, _) in enumerate(train_loader, 1):
        net.zero_grad()
        data.data.resize_(images.size()).copy_(images)
        output, _, mean, std = net(data)
        loss = loss_f(output, data, mean, std)
        loss.backward()
        optimizer.step()

        if step % 200 == 0:
            net.eval()
            eps = Variable(inputs)
            if torch.cuda.is_available():
                eps = eps.cuda()
            output= net(eps)[0]

            viz.images(output[:16].data.cpu().view(-1, 1, 28, 28), win=image, nrow=8)
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, step * len(data), len(train_loader.dataset),
                       100. * step / len(train_loader),
                       loss.data[0]/BATCH_SIZE))
            # if step == 200:
            #    viz.images(output[:16].data.cpu().view(-1, 1, 28, 28), nrow=8 ,opts=dict(title="epoch:{}".format(epoch)))
               # viz.scatter(X=tags.data.cpu(), Y=labels + 1, win=scatter, opts=dict(showlegend=True))

if hidden_size == 3:
    for step, (images, labels) in enumerate(test_loader, 1):
        if step > 1:
            break
        if torch.cuda.is_available():
            images = images.cuda()
        images = Variable(images)
        mean, std = net.encoder(images)
        tags = net.sampler(mean, std)
        viz.scatter(X=tags.data.cpu(), Y=labels + 1, win=scatter, opts=dict(legend=[str(a) for a in range(10)]
))