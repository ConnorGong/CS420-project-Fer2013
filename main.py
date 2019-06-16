from modules.model import ResNet18,Model
from modules.dataset import getDataSet
from modules.autologger import Logger

import os, time,shutil,sys
import matplotlib.pyplot as plt
import itertools
import pickle
import imageio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import random
from random import randrange as randr
from PIL import Image
import math


timeStr=time.strftime('%y%m%d-%H%M%S')
print(timeStr)
folder_path='rec{}/'.format(timeStr)
if not os.path.exists(folder_path):
    os.mkdir(folder_path)
sys.stdout=Logger(folder_path+'nn-b-'+timeStr+'.txt')
img_size = 48
batch_size = 128
num_epoch = 250
use_cuda=True
learning_rate_decay_start = 80  # 50
learning_rate_decay_every = 5 # 5
learning_rate_decay_rate = 0.9 # 0.9

device=torch.device("cuda")# if use_cuda else torch.device("cpu")

train_transform = transforms.Compose([
    transforms.RandomCrop(44),
    transforms.RandomHorizontalFlip(),
    #transforms.Grayscale(),
    transforms.ToTensor(),
    #transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5))
])
test_transform = transforms.Compose([
    transforms.RandomCrop(44),
    #transforms.Grayscale(),
    transforms.ToTensor(),
    #transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5))
])
train_set=getDataSet(train_transform,"datasets/train/")
test_set=getDataSet(test_transform,"datasets/test/")
train_loader=DataLoader(train_set,batch_size=batch_size,shuffle=True)
test_loader=DataLoader(test_set,batch_size=batch_size,shuffle=True)

model=ResNet18(num_classes=7).to(device)
#model=Model().to(device)
print(model)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer,learning_rate_decay_rate)
def train(epoch):
    #train
    torch.enable_grad()
    #global real_sample
    train_print_interval=100
    model.train()
    if epoch>learning_rate_decay_start and learning_rate_decay_start>0 and \
        (epoch-learning_rate_decay_start)%learning_rate_decay_every==0:
        scheduler.step()
    for batch_id,(x_, y_) in enumerate(train_loader):
        optimizer.zero_grad()
        #print(batch_id,'#',x_.shape,y_)
        #x_=x_+torch.tensor(np.random.laplace(0,0.01,x_.shape),dtype=x_.dtype)
        x_,y_ = x_.to(device),y_.to(device)
        D_result = model(x_)
        D_loss=F.cross_entropy(D_result,y_)
       # D_loss=F.nll_loss(D_result,y_)
        #print(batch_id,'#',D_result,y_,D_loss)
        D_loss.backward()
        optimizer.step()
        if batch_id%train_print_interval==0:
            print("epoch#{} D_train batch#{}/{} loss={}".format(
                epoch,batch_id,len(train_loader),D_loss.cpu().item())
            )

def test(epoch):
    test_loss = 0
    correct = 0
    cnt=0
    model.eval()
    with torch.no_grad():
        for batch_id, (data, target) in enumerate(test_loader):
            data,target = data.to(device), target.to(device)
            output=model(data)
            test_loss += F.cross_entropy(output, target, reduction="sum").item() # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            target,pred =target.cpu(),pred.cpu()
            correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
            cnt+=target.size()[0]
    test_loss /= cnt
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, cnt,100. * correct / cnt))
    return 1. * float(correct) / cnt

start_time = time.time()
for epoch in range(num_epoch):
    epoch_start_time = time.time()
    print("----- epoch #{} -----".format(epoch))
    train(epoch)
    accuracy=test(epoch)
    epoch_end_time = time.time()
    if epoch%50==0:
        torch.save(model.state_dict(),folder_path+'model_e{}_t{}.pth'.format(epoch,timeStr))
    per_epoch_ptime = epoch_end_time - epoch_start_time
    print('epoch #%d/%d - ptime: %.2f' % (epoch , num_epoch, per_epoch_ptime))

end_time = time.time()
total_ptime = end_time - start_time

sys.stdout.stop()
