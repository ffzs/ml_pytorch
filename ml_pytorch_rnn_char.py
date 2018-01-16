#coding:utf-8
"""
ffzs
2018.1.15
win10
i7-6700HQ
GTX965M
"""
import torch
from torch import nn, optim
from torch.autograd import Variable
import visdom
import numpy as np
import time
import glob
import unicodedata
import string
import random

USE_GPU = False

if USE_GPU:
    gpu_status = torch.cuda.is_available()
else:
    gpu_status = False

# string.ascii_letters生成所有字母， string.digits 生成数字
all_letters = string.ascii_letters+".,:'"
n_letters = len(all_letters)

# unicode 转 标准ASCII编码
def unicode_to_ascii(s):
    s = "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != 'Mn'and c in all_letters)
    return s

def readline(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicode_to_ascii(line) for line in lines]

category_lines = {}
all_categories = []
filenames = glob.glob('data/names/*.txt')
for filename in filenames:
    category = filename.split("\\")[-1].split(".")[0]
    all_categories.append(category)
    category_lines[category] = readline(filename)
n_categories = len(all_categories)
# print(categroy_lines['Chinese'])

def letter_to_index(letter):
    return all_letters.find(letter)

def letter_to_tensor(letter):
    tensor = torch.zeros(1,n_letters)
    tensor[0][letter_to_index(letter)] = 1
    return tensor

def line_to_tensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for i, letter in enumerate(line):
        tensor[i][0][letter_to_index(letter)] = 1
    return tensor

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.in2h = nn.Linear(input_size+hidden_size, hidden_size)
        self.in2o = nn.Linear(input_size+hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.in2h(combined)
        output = self.in2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        if gpu_status:
            return Variable(torch.ones(1, self.hidden_size)).cuda()
        else:
            return Variable(torch.ones(1, self.hidden_size))

n_hidden = 128
net = RNN(n_letters, n_hidden, n_categories)

if gpu_status:
    net = net.cuda()
    print("#"*30, '使用GPU', '#'*30)

def category_from_output(output):
    _ , top = output.data.topk(1)
    categroy_i = top[0][0]
    return all_categories[categroy_i], categroy_i

def random_choice(categories):
    return categories[random.randint(0, len(categories)-1)]

def random_train_example():
    category = random_choice(all_categories)
    line = random_choice(category_lines[category])
    category_tensor = Variable(torch.from_numpy(np.array([all_categories.index(category)])).long())
    line_tensor = Variable(line_to_tensor(line))
    return category, line, category_tensor, line_tensor


category, line, category_tensor, line_tensor = random_train_example()

loss_f = nn.NLLLoss()

def train(category, category_tensor, line_tensor):
    hidden = net.initHidden()
    net.zero_grad()
    for i in range(line_tensor.size()[0]):
        output, hidden = net(line_tensor[i], hidden)

    loss = loss_f(output, category_tensor)
    loss.backward()

    for p in net.parameters():
        # learning_rate
        p.data.add_(-0.005, p.grad.data)
    pred, _ = category_from_output(output)
    correct = 1 if pred == category else 0

    return output, loss.data[0], correct

net.load_state_dict(torch.load('save/char.pth'))

start_time = time.time()
running_loss, running_acc = 0., 0.
steps = 100000
print_every = 5000

# for step in range(1, steps+1):
#     category, _, category_tensor, line_tensor = random_train_example()
#     if gpu_status:
#         category_tensor = category_tensor.cuda()
#         line_tensor = line_tensor.cuda()
#     output, loss, correct = train(category, category_tensor, line_tensor)
#     running_loss += loss
#     running_acc += correct
#
#     if step % print_every == 0:
#         print("{} | {:.1f}% | loss: {:.4f}| acc: {:.2f}%| time: {:.2f}".format(step,step/steps*100,running_loss/print_every,
#                                                                 running_acc/print_every*100,
#                                                                time.time()-start_time))
#
#         running_loss, running_acc = 0., 0.

def evaluate(line_tensor):
    hidden = net.initHidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = net(line_tensor[i], hidden)

    return output

confusion = torch.zeros(n_categories, n_categories)

for category in all_categories:
    for line in category_lines[category]:
        line_tensor = Variable(line_to_tensor(line))
        output = evaluate(line_tensor)
        guess, guess_i = category_from_output(output)
        category_i = all_categories.index(category)
        confusion[category_i][guess_i] += 1
for i in range(n_categories):
    confusion[i] = confusion[i] / confusion[i].sum()

viz = visdom.Visdom()

viz.heatmap(X=confusion, opts=dict(
    columnnames=all_categories,
    rownames=all_categories,
    colormap="Jet",
    xlabel='guess',
    ylabel='category',
    marginleft=100,
    marginbottom=100,
))

torch.save(net.state_dict(), 'save/char.pth')
