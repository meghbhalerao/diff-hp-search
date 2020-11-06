import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import os
import argparse
from loaders.dataset import *
from archs.net import * 

parser = argparse.ArgumentParser(description='Train model with')


transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_val = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

train_data = Imagelists(image_list = os.path.join("./data/cifar/lists/train.txt"),root="./data/cifar/train/", transform=transform_train)
val_data = Imagelists(image_list = os.path.join("./data/cifar/lists/val.txt"),root="./data/cifar/train/", transform=transform_val)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True, num_workers=2)

val_loader = torch.utils.data.DataLoader(val_data, batch_size=100, shuffle=False, num_workers=2)

net = AlexNetBase()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

# Definfing the hyperparameter A's weights
A = torch.ones(len(train_loader.dataset))
A.requires_grad = True
print(A.shape)