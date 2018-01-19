"""
ffzs
2018.1.12
win10
i7-6700HQ
GTX965m
"""
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import transforms, datasets
import time
import visdom

viz = visdom.Visdom()

EPOCHS = 20
BATCH_SIZE = 64
LR = 0.005

USE_GPU = True

gpu_status = torch.cuda.is_available() if USE_GPU else False

train_dataset = datasets.MNIST('./mnist', True, transforms.ToTensor(), download=False)
test_dataset = datasets.MNIST('./mnist', False, transforms.ToTensor())

train_loader = DataLoader(train_dataset, BATCH_SIZE, True)

test_data = test_dataset.test_data[:2000]
test_label =test_dataset.test_labels[:2000]
# 数据可视化
viz.images(torch.unsqueeze(test_data[:25], 1), nrow=5)

if gpu_status:
    test_data = test_data.cuda()

test_data = Variable(test_data, volatile=True).float()/255.

class RNN(nn.Module):
    def __init__(self, in_dim, n_class):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(
            # 输入28 ，输出 64
            input_size=in_dim,
            hidden_size=64,
            # 神经层 2 层
            num_layers=2,
            # out(batch, 序列长度, 维度)
            batch_first=True)
        # n_class为输出的分类数
        self.cf = nn.Linear(64, n_class)

    def forward(self, x):
        # RNN(LSTM)的输出是output 和 hidden 这里只取 output
        out = self.rnn(x)[0]
        # 由于LSTM是根据具有对序列的记忆能力，我们只输出序列的最后一位，来判断
        out = out[:, -1, :]
        out = self.cf(out)
        return out

net = RNN(28, 10)

if gpu_status:
    net = net.cuda()
    print("#"*30, "使用gpu", "#"*30)
else:
    print("#" * 30, "使用cpu", "#" * 30)
loss_f = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=LR)

start_time = time.time()
for epoch in range(EPOCHS):
    sum_loss, sum_acc, sum_step = 0., 0., 0.
    for i, (tr_x, tr_y) in enumerate(train_loader, 1):
        tr_x = tr_x.view(-1, 28, 28)
        if gpu_status:
            tr_x, tr_y = tr_x.cuda(), tr_y.cuda()
        tr_x, tr_y = Variable(tr_x), Variable(tr_y)
        tr_out = net(tr_x)
        loss = loss_f(tr_out, tr_y)
        sum_loss += loss.data[0]*len(tr_y)
        pred_tr = torch.max(tr_out, 1)[1]
        sum_acc += sum(pred_tr==tr_y).data[0]
        sum_step += len(tr_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 40 == 0:
            net.eval()
            ts_out = net(test_data)
            pred_ts = torch.max(ts_out, 1)[1].cpu().data
            acc_ts = sum(pred_ts==test_label)/float(test_label.size(0))
            print("epoch: [{}/{}] | Loss: {:.4f} | TR_acc: {:.4f} | TS_acc: {:.4f} | Time: {:.1f}".format(epoch + 1, EPOCHS,
                                                     sum_loss / sum_step, sum_acc / sum_step, acc_ts, time.time() - start_time))
            sum_loss, sum_acc, sum_step = 0., 0., 0.




