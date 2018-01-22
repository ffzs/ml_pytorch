"""
ffzs
2018.1.20
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

TRAIN = True
LOAD_SAVE = False

viz = visdom.Visdom()

# string.ascii_letters生成所有字母， string.digits 生成数字
all_letters = string.ascii_letters+" .,;'-"
# 因为有EOS 所以多一个
n_letters = len(all_letters)+1

# unicode 转 标准ASCII编码
def unicode_to_ascii(s):
    s = "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != 'Mn'and c in all_letters)
    return s

# 输入文件获取名字信息
def readline(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicode_to_ascii(line) for line in lines]

category_lines = {}
all_categories = []
# 获取文件夹中所有.txt 文件名
filenames = glob.glob('data/names/*.txt')

# {'国家':'[名字]'}的格式所有数据， [国家] 的形式存储label
for filename in filenames:
    category = filename.split("\\")[-1].split(".")[0]
    all_categories.append(category)
    category_lines[category] = readline(filename)
n_categories = len(all_categories)

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.in2h = nn.Linear(n_categories+input_size+hidden_size, hidden_size)
        self.in2o = nn.Linear(n_categories+input_size+hidden_size, output_size)
        self.o2o = nn.Linear(hidden_size+output_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, category, input, hidden):
        input_combined = torch.cat((category, input, hidden), 1)
        hidden = self.in2h(input_combined)
        output = self.in2o(input_combined)
        output_conbined = torch.cat((hidden, output), 1)
        output = self.o2o(output_conbined)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return Variable(torch.zeros(1, self.hidden_size))

n_hidden = 128
net = RNN(n_letters, n_hidden, n_letters)

# 随机获取列表中的一个单位
def random_choice(categories):
    return categories[random.randint(0, len(categories)-1)]

# 随机获取一个国家和一个信息
def random_train_pair():
    category = random_choice(all_categories)
    line = random_choice(category_lines[category])
    return category, line

# 对国家标签进行 one-hot 处理
def category2tensor(category):
    li = all_categories.index(category)
    tensor = torch.zeros(1, n_categories)
    tensor[0][li] = 1
    return tensor

# 对line的每一个字符做one-hot处理
def input2tensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li in range(len(line)):
        letter = line[li]
        tensor[li][0][all_letters.find(letter)] = 1
    return tensor
# 在每一个字符串后面加一个结束标志
def target2tensor(line):
    letter_indexs = [all_letters.find(line[li]) for li in range(1, len(line))]
    letter_indexs.append(n_letters - 1)
    return torch.LongTensor(letter_indexs)

# 随机选取国家进行训练
def random_train_example():
    category, line = random_train_pair()
    category_tensor = Variable(category2tensor(category))
    input_line_tensor = Variable(input2tensor(line))
    target_line_tensor = Variable(target2tensor(line))
    return category_tensor, input_line_tensor, target_line_tensor

loss_f = nn.NLLLoss()

def train(category_tensor, input_line_tensor, target_line_tensor):
    hidden = net.initHidden()
    net.zero_grad()
    loss = 0
    correct = 0

    for i in range(input_line_tensor.size()[0]):
        output, hidden = net(category_tensor, input_line_tensor[i], hidden)
        loss += loss_f(output, target_line_tensor[i])
        pred = torch.max(output, 1)[1].data[0]
        if pred == target_line_tensor[i].data[0]:
            correct += 1

    loss.backward()

    for p in net.parameters():
        p.data.add_(-0.0003, p.grad.data)

    return output, loss.data[0]/input_line_tensor.size(0), correct

if LOAD_SAVE:
    net.load_state_dict(torch.load('save/char_make.pth'))

if TRAIN:
    # 可视化loss， acc
    line1 = viz.line(Y=np.arange(1, 10, 2))
    time.sleep(0.5)
    line2 = viz.line(Y=np.arange(1, 10, 2))
    start_time = time.time()
    steps = 200000
    all_losses, all_accuracy = [], []
    total_loss = 0
    accuracy = 0.
    correct_num = 0
    for step in range(1, steps+1):
        output, loss, correct = train(*random_train_example())
        total_loss += loss
        if correct >= 4 :
            correct_num += 1

        if step % 500 == 0:
            # 记录数据点可视化绘图
            accuracy = correct_num/500.00
            all_losses.append(total_loss/500)
            all_accuracy.append(accuracy)
            viz.line(Y=np.array(all_losses), win=line1, opts=dict(title="loss"))
            viz.line(Y=np.array(all_accuracy), win=line2, opts=dict(title="acc"))

            if step % 5000 == 0:
                print("{} | {:.1f}% | loss: {:.4f}| acc: {:.2f}%| time: {:.2f}".format(step, step/steps*100, loss,
                                                                        accuracy*100, time.time()-start_time))
            total_loss = 0
            correct_num = 0

# 根据给的第一个字母生成一个字符串
def sample(category, start_letter='A'):
    category_tensor = Variable(category2tensor(category))
    input = Variable(input2tensor(start_letter))
    hidden = net.initHidden()
    output_name = start_letter

    # 20只是一个范围而已，训练的还算可以的话不会出现这么长的，当然你也可以定的更大一些
    for i in range(20):
        output, hidden = net(category_tensor, input[0], hidden)
        index = torch.max(output, 1)[1].data[0]
        # 循环得到字符串终止提示后break 终止 for 循环
        if index == n_letters - 1:
            break
        # 不是终止信号则 将输出继续传递，直到出现终止
        else:
            letter = all_letters[index]
            output_name += letter
        input = Variable(input2tensor(letter))

    return output_name

def samples(category, start_letters='ABC'):
    for start_letter in start_letters:
        out = sample(category, start_letter)
        sign = '✗'
        if out in category_lines[category]:
            sign = '✓'
        # print(category, out, sign)
        return sign

# 每个国家随机选取1000个样本进行测试，收集数据绘制直方图
statistics = torch.zeros(len(all_categories))
for i in range(len(all_categories)):
    category = all_categories[i]
    for _ in range(1000):
        line = random_choice(category_lines[category])
        sign = samples(category,line[0])
        if sign == '✓' :
            statistics[i] += 1
viz.bar(X=statistics, opts=dict(rownames=all_categories))

# torch.save(net.state_dict(), 'save/char_make.pth')
