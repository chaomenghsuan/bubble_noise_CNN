
#/usr/bin/python3

import os
import re
import itertools
import torch
import torchvision
import torch.nn as nn
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

#########################
#### Load the data ######
#########################

gpu = torch.cuda.is_available()
if gpu:
    print('working on GPU')

path = ['/home/mzhao/bubble_backup/sliu-zhe_shi_ma_zi_result/trim=00,length=0,win_ms=064/feat/bpssliu', 
'/home/mzhao/bubble_backup/czhou-zhe_shi_ma_zi_result/trim=00,length=0,win_ms=064/feat/bpsczhou', 
'/home/mzhao/bubble_backup/gge-zhe_shi_ma_zi_result/trim=00,length=0,win_ms=064/feat/bpsgge', 
'/home/mzhao/bubble_backup/yguo-zhe_shi_ma_zi_result/trim=00,length=0,win_ms=064/feat/bpsyguo',
'/home/mzhao/bubble_backup/qxu-zhe_shi_ma_zi_result/trim=00,length=0,win_ms=064/feat/bpsqxu']
f1 = [sorted([f for f in os.listdir(path[i]) if re.match(r'^ma3.+\.mat$', f)]) for i in range(len(path))]

def read_mat(file_name_list, path_list):
    assert len(file_name_list) == len(path_list)
    assert type(file_name_list) == type(path_list) == list
    for i in range(len(path_list)):
        for file in file_name_list[i]:
            path = os.path.join(path_list[i], file)
            yield loadmat(path)

loadma1 = tuple(read_mat(f1, path))

def reshape(dataset):
    for mat in dataset:
        yield (mat['features'].reshape((mat['origShape'][0][1], mat['origShape'][0][0]))).T

ma1 = tuple(reshape(loadma1))
ma1 = np.array(ma1)
#print(ma1.shape)

########################
##### Load the labels ##
########################

result_file = ['/home/mzhao/bubble_backup/sliu-zhe_shi_ma_zi/sliu.csv',
'/home/mzhao/bubble_backup/czhou-zhe_shi_ma_zi/czhou.csv',
'/home/mzhao/bubble_backup/gge-zhe_shi_ma_zi/gge.csv',
'/home/mzhao/bubble_backup/yguo-zhe_shi_ma_zi/yguo.csv',
'/home/mzhao/bubble_backup/qxu-zhe_shi_ma_zi/qxu.csv']
csv = [pd.read_csv(result_file[i]) for i in range(len(result_file))]

labels1 = [csv[i].loc[csv[i]['Input.rightAnswer1'] == 'ma3'] for i in range(len(result_file))]
labels1 = [list((labels1[i]['Input.rightAnswer1'] == labels1[i]['Answer.wordchoice1']).astype(float)) 
           for i in range(len(result_file))]
labels1 = np.array(list(itertools.chain.from_iterable(labels1)))
#print(labels1.shape)

########################
#### Pre-processing ####
########################

ma1_test = ma1[::10]
ma1_train = np.array([ma1[i] for i in range(len(ma1)) if i%10 != 0])
l1_test = labels1[::10]
l1_train = np.array([labels1[i] for i in range(len(labels1)) if i%10 != 0])
ma1_dev = ma1_test[::2]
l1_dev = l1_test[::2]
ma1_test = np.array([ma1_test[i] for i in range(len(ma1_test)) if i%2 != 0])
l1_test = np.array([l1_test[i] for i in range(len(l1_test)) if i%2 != 0])

train_size = ma1_train.shape[0]
dev_size = ma1_dev.shape[0]
test_size = ma1_test.shape[0]
print('train:', train_size, 'dev:', dev_size, 'test:', test_size)

X_train = torch.from_numpy(ma1_train).float().view(train_size, 1, 1412, 121)
X_dev = torch.from_numpy(ma1_dev).float().view(dev_size, 1, 1412, 121)
X_test = torch.from_numpy(ma1_test).float().view(test_size, 1, 1412, 121)

y_train = torch.from_numpy(l1_train).float()
y_dev = torch.from_numpy(l1_dev).float()
y_test = torch.from_numpy(l1_test).float()

print('trainning set baseline:', np.sum(np.array(y_train))/train_size)
print('dev set baseline:', np.sum(np.array(y_dev))/dev_size)
print('testing set baseline:', np.sum(np.array(y_test))/test_size)

batch = 50

def make_batch(data, batch_size):
    assert len(data)%batch_size == 0
    for i in range(0, len(data), batch_size):
        yield data[i:i+batch_size]

def data_label_zip(dataset, labelset):
    return zip(dataset, labelset)

X_train_b = tuple(make_batch(X_train, batch))
X_test_b = tuple(make_batch(X_test, batch))
X_dev_b = tuple(make_batch(X_dev, batch))
y_dev_b = tuple(make_batch(y_dev, batch))
y_train_b = tuple(make_batch(y_train, batch))
y_test_b = tuple(make_batch(y_test, batch))

#train, test = data_label_zip(X_train_b, y_train_b), data_label_zip(X_test_b, y_test_b)

########################
#### CNN frame #########
########################

class ConvNN(nn.Module):
    def __init__(self):
        super(ConvNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=15, stride=1, padding=6), # 1410 * 119
            nn.ReLU(),
            nn.Conv2d(4, 8, kernel_size=11, stride=1, padding=4), #1408 * 117
            nn.ReLU(),
#1408 = 2 * 2 * 2 * 2 * 2 * 2 * 2 * 11
#117 = 3 * 3 * 13
            nn.MaxPool2d(kernel_size=(16,9))) # 88*39
        self.layer2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=7, stride=1, padding=3),
            nn.Tanh(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=(2,1))) # 44 * 39
        self.drop_out = nn.Dropout(p=0.4)
        self.fc1 = nn.Linear(44 * 13 * 32, 2)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()

    def forward(self, x):
        out = self.layer1(x)
        #out = self.drop_out(out)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        #out = self.sigmoid(out)
        #out = self.softmax(out)
        out = self.drop_out(out)
        return out

cnn = ConvNN()
if gpu:
    cnn.cuda()

#for p in cnn.parameters():
#    print(p.shape)

num_epochs = 30
#num_classes = 2
learning_rate = 0.0001

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
#criterion = nn.L1Loss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)

########################
## Train the network ###
########################

for i in range(num_epochs):
    correct = 0
    loss_sum = 0
    for j, (x, y) in enumerate(data_label_zip(X_train_b, y_train_b)):
        if gpu:
            x, y = x.cuda(), y.cuda()
        outputs = cnn(x)
        loss = criterion(outputs, y.long())
        
        # Backprop and perform Adam optimisation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track the accuracy
        if gpu:
            #pred = torch.max(outputs.data, 1)[1].cuda()
            pred = outputs.data.max(1)[1].cuda()
        else:
            #_, pred = torch.max(outputs.data, 1)
            _, pred = outputs.data.max(1)
        #print('output:', outputs.data, 'prediction:', pred)
        correct += (pred == y.long()).sum().item()
        loss_sum += loss.item()
        if (j+1) % (len(X_train)/batch) == 0:
            print('Loss:',loss_sum,'Accuracy for epoch %s:' %(i+1), correct / train_size)
            print('Epoch %s done' %(i+1))

cnn.eval()
with torch.no_grad():
    correct_dev = 0
    correct_test = 0
    pred_dev = []
    pred_test = []
    for x, y in data_label_zip(X_dev_b, y_dev_b):
        if gpu:
            x, y = x.cuda(), y.cuda()
        outputs = cnn(x)
        if gpu:
            pred = outputs.data.max(1)[1].cuda()
        else:
            _, pred = outputs.data.max(1)
        correct_dev += (pred == y.long()).sum().item()
        pred_dev.append(pred)
    pred_dev = list(itertools.chain.from_iterable(pred_dev))
    
    for x, y in data_label_zip(X_test_b, y_test_b):              
        if gpu:                                                
            x, y = x.cuda(), y.cuda()                          
        outputs = cnn(x)                                       
        if gpu:                                                
            pred = outputs.data.max(1)[1].cuda()               
        else:                                                  
            _, pred = outputs.data.max(1)                                                           
        correct_test += (pred == y.long()).sum().item()         
        pred_test.append(pred)
    pred_test = list(itertools.chain.from_iterable(pred_test))

    print('Dev set baseline:', np.sum(np.array(y_dev)) / dev_size, 'Dev set accuracy:', correct_dev / dev_size)
    precision_dev = precision_score(y_dev, pred_dev, average='macro')
    recall_dev = recall_score(y_dev, pred_dev, average='macro')
    f1_dev = f1_score(y_dev, pred_dev, average='macro')
    cfm_dev = confusion_matrix(y_dev, pred_dev)
    cfm_dev = pd.DataFrame(cfm_dev, columns=['pred_0','pred_1'], index=['class_0','class_1'])
    
    print('Test set baseline:', np.sum(np.array(y_test)) / test_size, 'Test set accuracy:', correct_test / test_size)
    precision_test = precision_score(y_test, pred_test, average='macro')        
    recall_test = recall_score(y_test, pred_test, average='macro')
    f1_test = f1_score(y_test, pred_test, average='macro')        
    cfm_test = confusion_matrix(y_test, pred_test)                
    cfm_test = pd.DataFrame(cfm_test, columns=['pred_0','pred_1'], index=['class_0','class_1'])

    print('=====development set result=====')
    print('development set precision:', precision_dev)
    print('development set recall:', recall_dev)
    print('development set F1 score:', f1_dev)
    print(cfm_dev)

    print('=====test set result=====')                  
    print('development set precision:', precision_test)          
    print('development set recall:', recall_test)               
    print('development set F1 score:', f1_test)                 
    print(cfm_test)  
