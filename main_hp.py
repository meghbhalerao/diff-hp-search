import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import os
import argparse
from loaders.dataset import *
from archs.net import * 
from utils.losses import *

parser = argparse.ArgumentParser(description='Train model with')

def main():
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


    f = open("./data/cifar/lists/train.txt","r")
    f_list = [line for line in f]
    weight_sample = torch.ones(len(f_list))
    train_data = Imagelists(image_list = os.path.join("./data/cifar/lists/train.txt"),weight_sample = weight_sample, root="./data/cifar/train/", transform=transform_train)
    val_data = Imagelists(image_list = os.path.join("./data/cifar/lists/val.txt"), weight_sample = weight_sample, root="./data/cifar/train/", transform=transform_val)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True, num_workers=2)

    val_loader = torch.utils.data.DataLoader(val_data, batch_size=100, shuffle=False, num_workers=2)

    model = AlexNet().cuda()

    criterion = CE_Mask().cuda()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    num_epochs = 100
    learning_rate_min =  0.001
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(num_epochs), eta_min=learning_rate_min)

    for epoch in range(num_epochs):
        train(model, train_loader, val_loader, criterion)
        val()
        scheduler.step()

def train(model, train_loader, val_loader, criterion):
    for idx, (img, targets, weight, _) in enumerate(train_loader):
        logits = model(img.cuda())
        print(criterion(logits,targets))
    return 0

def val():
    pass

if __name__ == "__main__":
    main()





















#for att in dir(train_loader.dataset):
#    print(att,getattr(train_loader.dataset, att))
# Definfing the hyperparameter A's weights 