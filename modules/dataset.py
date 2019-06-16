import torch
from torchvision import datasets, transforms

def getDataSet(transform,path="datasets/train/"):
    data_set=datasets.ImageFolder(root=path,transform=transform)
    return data_set
