# -*- 本脚本用于电影评论集情感分析 Created by Songyujian 2023.7.27-*-
import pandas as pd
from collections import Counter
import os
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from gensim.models import keyedvectors
import visdom
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

# 数据读取
root_path = "/Users/songyujian/Downloads/Dataset/"
train_path = "/Users/songyujian/Downloads/Dataset/train.txt"

train_data = pd.read_csv("/Users/songyujian/Downloads/Dataset/train.txt", names=["label", "comment"], sep="\t")
# 单个评论词数（数据集已分词）
comments_len = train_data.iloc[:, 1].apply(lambda x: len(x.split()))
train_data["comments_len"] = comments_len
train_data["comments_len"].describe(percentiles=[.5, .95])

# 词列表
words = []
for i in range(len(train_data)):
    com = train_data["comment"][i].split()
    words = words + com

# 保留出现频率大于25的词，存入 word_freq.txt
Freq = 25
with open(os.path.join(root_path, "word_freq.txt"), 'w', encoding='utf-8') as fout:
    for word, freq in Counter(words).most_common():
        if freq > Freq:
            fout.write(word + "\n")

# 将上述词用于初始vocab
with open(os.path.join(root_path, "word_freq.txt"), encoding='utf-8') as fin:
    vocab = [i.strip() for i in fin]
vocab = set(vocab)
word2idx = {i: index for index, i in enumerate(vocab)}
idx2word = {index: i for index, i in enumerate(vocab)}
vocab_size = len(vocab)
# 将出现频率低于25的词改为中性词
pad_id = word2idx["把"]

sequence_length = 62


# 对输入数据预处理，对句子用索引表示，对句子进行截断与padding，使用"把"来填充
def tokenizer():
    inputs = []
    sentence_char = [i.split() for i in train_data["comment"]]
    # 将输入文本进行padding
    for index, i in enumerate(sentence_char):
        # 词表中如果没有该稀有词，默认返回 pad_id
        temp = [word2idx.get(j, pad_id) for j in i]
        if (len(i) < sequence_length):
            # padding
            for _ in range(sequence_length - len(i)):
                temp.append(pad_id)
        else:
            temp = temp[:sequence_length]
        inputs.append(temp)
    return inputs


data_input = tokenizer()

# 准备训练
device = "cpu"
Embedding_size = 50
Batch_Size = 32
# 卷积核的数量
Kernel = 3
Filter_num = 10
Epoch = 80
Dropout = 0.5
Learning_rate = 1e-3


# 数据集划分
class TextCNNDataSet(Data.Dataset):
    def __init__(self, data_inputs, data_targets):
        self.inputs = torch.LongTensor(data_inputs)
        self.label = torch.LongTensor(data_targets)

    def __getitem__(self, index):
        return self.inputs[index], self.label[index]

    def __len__(self):
        return len(self.inputs)


TextCNNDataSet = TextCNNDataSet(data_input, list(train_data["label"]))
# 训练规模
train_size = int(len(data_input) * 0.8)
# 测试规模
test_size = int(len(data_input) * 0.05)
val_size = len(data_input) - train_size - test_size
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(TextCNNDataSet,
                                                                         [train_size, val_size, test_size])

TrainDataLoader = Data.DataLoader(train_dataset, batch_size=Batch_Size, shuffle=True)
TestDataLoader = Data.DataLoader(test_dataset, batch_size=Batch_Size, shuffle=True)

# 加载word2vec
w2v = keyedvectors.load_word2vec_format(os.path.join(root_path, "wiki_word2vec_50.bin"), binary=True)
# 初始化visdom
viz = visdom.Visdom()


# 使用word2vec版本
def word2vec(x):
    x2v = np.ones((len(x), x.shape[1], Embedding_size))
    for i in range(len(x)):
        x2v[i] = w2v[[idx2word[j.item()] for j in x[i]]]
    return torch.tensor(x2v).to(torch.float32)


# 2分类问题
num_classs = 2


# 定义TEXT-CNN网络
class TextCNN(nn.Module):
    def __init__(self):
        super(TextCNN, self).__init__()
        # 通道
        out_channel = Filter_num
        self.conv = nn.Sequential(
            nn.Conv2d(1, out_channel, (2, Embedding_size)),
            # 卷积核大小为2*Embedding_size，默认步长为1
            nn.ReLU(),
            nn.MaxPool2d((sequence_length - 1, 1)),
        )
        self.dropout = nn.Dropout(Dropout)
        self.fc = nn.Linear(out_channel, num_classs)

    def forward(self, X):
        batch_size = X.shape[0]
        embedding_X = word2vec(X)
        embedding_X = embedding_X.unsqueeze(1)
        conved = self.conv(embedding_X)
        conved = self.dropout(conved)
        flatten = conved.view(batch_size, -1)
        output = self.fc(flatten)
        # 二分类问题，使用softmax表示概率
        return F.log_softmax(output)

# 训练网络
model = TextCNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=Learning_rate)


def binary_acc(pred, y):
    """
    计算模型的准确率
    :param pred: 预测值
    :param y: 实际真实值
    :return: 返回准确率
    """
    correct = torch.eq(pred, y).float()
    acc = correct.sum() / len(correct)
    return acc.item()


def train():
    """
    模型训练
    optimizer: 使用的模型优化策略
    :return: 返回模型训练结果的平均准确值
    """
    avg_acc = []
    model.train()
    for index, (batch_x, batch_y) in enumerate(TrainDataLoader):
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        pred = model(batch_x)
        loss = F.nll_loss(pred, batch_y)
        acc = binary_acc(torch.max(pred, dim=1)[1], batch_y)
        avg_acc.append(acc)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    avg_acc = np.array(avg_acc).mean()
    return avg_acc

# 绘制训练准确率图
viz.line([0], [0], win='Train_acc', opts=dict(title="训练准确率"))
# 训练循环
model_train_acc, model_test_acc = [], []
for epoch in range(Epoch):
    train_acc = train()
    print("epoch = {}, 训练准确率={}".format(epoch + 1, train_acc))
    viz.line([train_acc], [epoch + 1], win='Train_acc', update='append')
    model_train_acc.append(train_acc)

# 绘制测试准确率图
viz.line([0], [0], win='Test_acc', opts=dict(title="测试准确率"))
# 定义评价模型
def evaluate():
    """
    模型评估
    :param model: 使用的模型
    :return: 返回当前训练的模型在测试集上的结果
    """
    avg_acc = []
    model.eval()  # 进入测试模式
    ev_step = 0
    with torch.no_grad():
        for x_batch, y_batch in TestDataLoader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            pred = model(x_batch)
            acc = binary_acc(torch.max(pred, dim=1)[1], y_batch)
            avg_acc.append(acc)
            viz.line([acc], [ev_step], win='Test_acc', update='append')
            ev_step += 1
    return np.array(avg_acc).mean()

evaluate()