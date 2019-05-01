import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random
import torchvision
import torchvision.transforms as transforms

class batch_reader(object):
    def __init__(self, dataset_path, train_ratio, dev_ratio):
        assert train_ratio + dev_ratio == 1
        self.train = []
        self.train_label = []
        self.val = []
        self.val_label = []
        with open(dataset_path, 'r', encoding='utf8') as f:
            for line in f:
                line = line.strip()
                line_seg = line.split(',')
                line_seg = [float(x) for x in line_seg]
                if random.random() < train_ratio:
                    self.train.append(line_seg[1:])
                    self.train_label.append(int(line_seg[0]))
                else:
                    self.val.append(line_seg[1:])
                    self.val_label.append(int(line_seg[0]))
    def get_next_batch(self, batch_size, train_mode):
        if train_mode:
            index = random.sample(range(len(self.train_label)), batch_size)
            data_batch = [self.train[i] for i in index]
            label_batch = [self.train_label[i] for i in index]
        else:
            index = random.sample(range(len(self.val_label)), batch_size)
            data_batch = [self.val[i] for i in index]
            label_batch = [self.val_label[i] for i in index]
        return data_batch, label_batch
class test_batch_reader(object):
    def __init__(self, dataset_path):
        #assert train_ratio + dev_ratio == 1
        self.train = []
        # #self.train_label = []
        # self.val = []
        # self.val_label = []
        with open(dataset_path, 'r', encoding='utf8') as f:
            for line in f:
                line = line.strip()
                line_seg = line.split(',')
                line_seg = [float(x) for x in line_seg]
                self.train.append(line_seg[:])
                # #if random.random() < train_ratio:
                #     self.train.append(line_seg[1:])
                #     self.train_label.append(int(line_seg[0]))
                # else:
                #     self.val.append(line_seg[1:])
                #     self.val_label.append(int(line_seg[0]))
    def get_next_batch(self, batch_size):
        index = random.sample(range(len(self.train)), batch_size)
        data_batch = [self.train[i] for i in index]

        # if train_mode:
        #     index = random.sample(range(len(self.train_label)), batch_size)
        #     data_batch = [self.train[i] for i in index]
        #     label_batch = [self.train_label[i] for i in index]
        # else:
        #     index = random.sample(range(len(self.val_label)), batch_size)
        #     data_batch = [self.val[i] for i in index]
        #     label_batch = [self.val_label[i] for i in index]
        return data_batch
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(13, 24),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(24, 24),
            nn.Linear(24, 2)
        )
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x
model = MLP()
print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()
mean_train_losses = []
mean_valid_losses = []
valid_acc_list = []
epochs = 5
batchreader = batch_reader(dataset_path='data2.csv', train_ratio=0.8, dev_ratio=0.2)

for epoch in range(epochs):
    train_losses = []
    valid_losses = []
    for i in range(1000):
        images, labels = batchreader.get_next_batch(batch_size=32*2, train_mode=True)
        optimizer.zero_grad()
        outputs = model.forward(torch.Tensor(images))
        loss = loss_fn(outputs, torch.LongTensor(labels))
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        if (i *32)%(32*100) == 0:
            print(f'{i*32}/50000')
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i in range(1000):
            images, labels = batchreader.get_next_batch(batch_size=32*2, train_mode=False)
            outputs = model.forward(torch.Tensor(images))
            loss = loss_fn(outputs, torch.LongTensor(labels))
            valid_losses.append(loss.item())
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == torch.LongTensor(labels)).sum().item()
            total += torch.LongTensor(labels).size(0)
    mean_train_losses.append(np.mean(train_losses))
    mean_valid_losses.append(np.mean(valid_losses))
    accuracy = 100 * correct / total
    print('epoch:{}, train loss:{:.4f}, valid loss:{:.4f}, valid acc:{:.2f}%'\
          .format(epoch+1, np.mean(train_losses), np.mean(valid_losses), accuracy))
model.eval()
#test_preds = torch.LongTensor()
testbatchreader = test_batch_reader(dataset_path='pred2.csv')
with torch.no_grad():
    #for i in range(1000):
    images = testbatchreader.get_next_batch(batch_size=1000)
    outputs = model.forward(torch.Tensor(images))
    _, predicted = torch.max(outputs.data, 1)
print(predicted)