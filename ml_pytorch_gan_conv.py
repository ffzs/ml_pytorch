"""
ffzs
2018.1.18
win10
i7-6700HQ
GTX965M
"""
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision import transforms, datasets
import visdom
from torch.autograd import Variable
import time

viz = visdom.Visdom()

BATCH_SIZE = 64
LR = 0.0002
EPOCHS = 100
fake_dim = 100
USE_GPU = True

gpu_status = torch.cuda.is_available() if USE_GPU else False

dataset = datasets.MNIST('./mnist', True, transform=transforms.Compose([transforms.ToTensor(),
                                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
# 加载一个batch数据
dataiter = iter(dataloader)
imgs, labels = dataiter.next()
# 可视化部分数据
viz.images(imgs[:64])
time.sleep(0.5)
images = viz.images(imgs[:64])

class Discrimination(nn.Module):
    def __init__(self):
        super(Discrimination, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 5, 1, 2),  # batch, 32, 28, 28
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d(2),  # batch, 32, 14, 14
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 5, 1, 2),  # batch, 64, 14, 14
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d(2)  # batch, 64, 7, 7
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 1024),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        #  x : batch, width, height, channel=1
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x.squeeze()

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc = nn.Linear(fake_dim, 1*56*56)
        self.br = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.ReLU(True)
        )
        self.deconv1 = nn.Sequential(
            nn.Conv2d(1, 50, 3, 1, 1),
            nn.BatchNorm2d(50),
            nn.ReLU(True)
        )
        self.deconv2 = nn.Sequential(
            nn.Conv2d(50, 25, 3, 1, 1),
            nn.BatchNorm2d(25),
            nn.ReLU(True)
        )
        self.deconv3 = nn.Sequential(
            nn.Conv2d(25, 1, 2, 2),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.fc(x)
        x = self.br(x.view(x.size(0), 1, 56, 56))
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        return x

net_d = Discrimination()
net_g = Generator()
gpu_info = "使用CPU"
if gpu_status:
    net_d.cuda()
    net_g.cuda()
    gpu_info = "使用GPU"

print("#"*30, gpu_info, "#"*30)

loss_f = nn.BCELoss()
optimizer_d = optim.Adam(net_d.parameters(), lr=LR)
optimizer_g = optim.Adam(net_g.parameters(), lr=LR)

for epoch in range(EPOCHS):
    for i,(imgs, _) in enumerate(dataloader, 1):
        imgs_num = imgs.size(0)
        real_imgs = Variable(imgs)
        real_label = Variable(torch.ones(imgs_num))
        fake_label = Variable(torch.zeros(imgs_num))
        fake_imgs = Variable(torch.randn(imgs_num, fake_dim))
        if gpu_status:
            real_imgs, real_label, fake_label, fake_imgs = real_imgs.cuda(), real_label.cuda(), \
                                                           fake_label.cuda(), fake_imgs.cuda()
        real_out = net_d(real_imgs)
        d_loss_real = loss_f(real_out, real_label)
        real_scores = real_out

        fake_img = net_g(fake_imgs)
        fake_out = net_d(fake_img)
        d_loss_fake = loss_f(fake_out, fake_label)
        fake_scores = fake_out
        d_loss = d_loss_fake + d_loss_real
        optimizer_d.zero_grad()
        d_loss.backward()
        optimizer_d.step()

        fake_imgs = Variable(torch.randn(imgs_num, fake_dim))
        if gpu_status:
            fake_imgs = fake_imgs.cuda()
        fake_img = net_g(fake_imgs)
        out = net_d(fake_img)
        loss = loss_f(out, real_label)

        optimizer_g.zero_grad()
        loss.backward()
        optimizer_g.step()

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], d_loss: {:.6f}, g_loss: {:.6f} '
                  'D real: {:.6f}, D fake: {:.6f}'.format(
                      epoch, EPOCHS, d_loss.data[0], loss.data[0],
                      real_scores.data.mean(), fake_scores.data.mean()))
        # if ( i+1 ) % 300 == 0:
            image = (fake_img[:64].cpu().data + 1)*0.5
            # print(image.view(-1, 1, 28, 28))
            viz.images(image.view(-1, 1, 28, 28), win=images)
        if i == 1:
            image = (fake_img[:64].cpu().data + 1) * 0.5
            viz.images(image.view(-1, 1, 28, 28),opts=dict(title="epoch:{}".format(epoch)))












