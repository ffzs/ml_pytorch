"""
ffzs
2018.1.11
win10
i7-6700HQ
gtx965m
"""
import time
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from visdom import Visdom
import numpy as np

viz = Visdom()

EPOCH = 2
BATCH_SIZE = 30
LEARNING_RATE = 0.001
UES_GPU = False

train_dataset = datasets.MNIST(root='./mnist', train=True,
                               transform=transforms.ToTensor(),download=False)
test_dataset = datasets.MNIST(root='./mnist', train=False, transform=transforms.ToTensor())

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# viz.image(train_dataset.train_data[2])

class Nueralnetwork(nn.Module):
    def __init__(self,in_dim,hidden1,hidden2,out_dim):
        super(Nueralnetwork, self).__init__()
        self.layer1 = nn.Linear(in_dim,hidden1)
        self.layer2 = nn.Linear(hidden1,hidden2)
        self.layer3 = nn.Linear(hidden2,out_dim)

    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

net = Nueralnetwork(28*28,200,100,10)

optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
loss_f = nn.CrossEntropyLoss()

if UES_GPU:
    gpu_status = torch.cuda.is_available()
    if gpu_status:
        net = net.cuda()
        print("#"*24,"使用gpu","#"*24)
    else:
        print("#"*24,"使用cpu","#"*24)
else:
    gpu_status = False
    print("#"*24,"使用cpu","#"*24)

# line = viz.line(Y=np.arange(1,3,1))
line2 = viz.line(Y=np.arange(1,3,1))
start_time = time.time()
test_acc = []
test_loss = []
epoches = []
loss_point = []
acc_point = []
time_point = []
for epoch in range(EPOCH):
    m_loss, m_acc = 0., 0.
    for num, (img, label) in enumerate(train_loader,1):
        img = img.view(img.size(0), -1)
        if gpu_status:
            tr_img = Variable(img).cuda()
            tr_label = Variable(label).cuda()
        else:
            tr_img = Variable(img)
            tr_label = Variable(label)
        out = net(tr_img)
        loss = loss_f(out, tr_label)
        m_loss += loss.data[0]*len(tr_label)
        pred = torch.max(out,1)[1]
        m_acc += sum(pred==tr_label).data[0]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if num % 30 == 0:
            print("  [{}/{}] | loss: {:.4f} | acc: {:.4f} | time: {:.1f} ".format(epoch+1,
                            EPOCH,m_loss/(num*BATCH_SIZE),
                            m_acc/(num*BATCH_SIZE), time.time()-start_time))
            loss_point.append(m_loss/(num*BATCH_SIZE))
            acc_point.append(m_acc/(num*BATCH_SIZE))
            time_point.append(time.time()-start_time)
            # viz.line(X=np.column_stack((np.array(time_point), np.array(time_point))),
            #          Y=np.column_stack((np.array(loss_point), np.array(acc_point))),
            #          win=line,
            #          opts=dict(legend=["Loss_tr", "Acc_tr"]))
    print("--TRAIN--Epoch: {} -|- loss: {:.4f} -|- acc: {:.4f} -|- time: {:.1f} --".format(epoch + 1,
                                 m_loss / len(train_dataset), m_acc / len(train_dataset),time.time() - start_time))
    net.eval()
    eval_loss = 0.
    eval_acc = 0.
    for num, (img, label) in enumerate(test_loader, 1):
        img = img.view(img.size(0), -1)
        if gpu_status:
            tt_img = Variable(img, volatile=True).cuda()
            tt_label = Variable(label, volatile=True).cuda()
        else:
            tt_img = Variable(img, volatile=True)
            tt_label = Variable(label, volatile=True)
        out = net(tt_img)
        loss = loss_f(out, tt_label)
        eval_loss += loss.data[0]*len(tt_label)
        pred = torch.max(out,1)[1]
        eval_acc += sum(pred==tt_label).data[0]
    epoches.append(epoch)
    test_acc.append(eval_acc/len(test_dataset))
    test_loss.append(eval_loss/len(test_dataset))
    # viz.line(X=np.column_stack((np.array(epoches), np.array(epoches))),
    #          Y=np.column_stack((np.array(test_loss), np.array(test_acc))),
    #          win=line2,
    #          opts=dict(legend=["Loss_tt", "Acc_tt"]))
    print("==TEST ==Epoch: {} =|= Loss: {:.4f} =|= Acc: {:.4f} =|= Time: {:.1f}==".format(epoch+1, eval_loss/len(test_dataset),
                                                                   eval_acc /len(test_dataset),time.time()-start_time))
print("TEST_ACC: {:.4f} | TEST_LOSS: {:.4f} | TIME: {:.1f}".format(np.array(test_acc).mean(), np.array(test_loss).mean(),
                                                                   time.time()-start_time))

# torch.save(net.state_dict(), './nn_state/neural_network.pth')







