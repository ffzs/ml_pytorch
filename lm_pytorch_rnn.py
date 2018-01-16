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

EPOCHS = 3
BATCH_SIZE = 64
LR = 0.001

USE_GPU = True

if USE_GPU:
    gpu_status = torch.cuda.is_available()
else:
    gpu_status = False

train_dataset = datasets.MNIST('./mnist', True, transforms.ToTensor(), download=False)
test_dataset = datasets.MNIST('./mnist', False, transforms.ToTensor())

train_loader = DataLoader(train_dataset, BATCH_SIZE, True)

test_data = test_dataset.test_data[:2000]
test_label =test_dataset.test_labels[:2000]

# viz.images(torch.unsqueeze(test_data[:25], 1), nrow=5)

if gpu_status:
    test_data = test_data.cuda()
test_data = Variable(test_data, volatile=True).float()/255. # channel 1

class RNN(nn.Module):
    def __init__(self, in_dim, n_class):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(
            input_size=in_dim,
            hidden_size=64,
            num_layers=2,
            batch_first=True)
        self.cf = nn.Linear(64, n_class)

    def forward(self, x):
        # x (batch, time_step ,input_size) out(batch,tiem_step, output_step)
        # (_,_) (n_layers, batch, hidden_size)
        out, (_,_ ) = self.rnn(x)
        out = out[:, -1, :]
        out = self.cf(out)
        return out

net = RNN(28,10)

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
        # forward
        tr_out = net(tr_x)
        loss = loss_f(tr_out, tr_y)
        sum_loss += loss.data[0]*len(tr_y)
        pred_tr = torch.max(tr_out, 1)[1]
        sum_acc += sum(pred_tr==tr_y).data[0]

        sum_step += len(tr_y)
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 40 == 0:
            ts_out = net(test_data)
            pred_ts = torch.max(ts_out, 1)[1].cpu().data
            acc_ts = sum(pred_ts==test_label)/float(test_label.size(0))
            print("epoch: [{}/{}] | Loss: {:.4f} | TR_acc: {:.4f} | TS_acc: {:.4f} | Time: {:.1f}".format(epoch + 1, EPOCHS,
                                                     sum_loss / sum_step, sum_acc / sum_step, acc_ts, time.time() - start_time))
            sum_loss, sum_acc, sum_step = 0., 0., 0.
torch.save(net.state_dict(), './save/rnn.pth')



