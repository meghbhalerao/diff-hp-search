
from os import access
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import math
import argparse
from loaders.dataset import *
from archs.net import * 
from utils.losses import *
from utils.operations import *
from copy import deepcopy

parser = argparse.ArgumentParser(description='Train model with')

def main():
  
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_ssl = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])


    f_train = open("./data/cifar/lists/train.txt","r")
    f_list = [line for line in f_train]
    A_train_bank = torch.ones(len(f_list))

    f_ssl = open("./data/cifar/lists/ssl.txt","r")
    f_list = [line for line in f_ssl]
    A_ssl_bank = torch.ones(len(f_list))

    ssl_data = Imagelists(image_list = os.path.join("./data/cifar/lists/ssl.txt"),weight_sample = A_ssl_bank, root="./data/cifar/train/", transform=transform_train)
    train_data = Imagelists(image_list = os.path.join("./data/cifar/lists/train.txt"),weight_sample = A_train_bank, root="./data/cifar/train/", transform=transform_train)
    val_data = Imagelists(image_list = os.path.join("./data/cifar/lists/val.txt"), weight_sample = None, root="./data/cifar/train/", transform=transform_val)

    bs = 100
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=bs, shuffle=True, num_workers=2, drop_last = True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=bs, shuffle=False, num_workers=2, drop_last = True)
    ssl_loader = torch.utils.data.DataLoader(ssl_data, batch_size=bs, shuffle=False, num_workers=2, drop_last = True)

    criterion = CE_Mask().cuda()
    criterion_ssl = CE_Mask().cuda()
    W = AlexNet(criterion = CE_Mask, num_classes = 10).cuda()
    H = Predictor_deep(criterion = CE_Mask, num_class=10).cuda()

    optimizer_H = optim.SGD(H.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    optimizer_W = optim.SGD(W.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

    num_iter = 50000
    learning_rate_min =  0.001
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_, float(num_iter/len(train_loader)), eta_min=learning_rate_min)
    best_val_acc = 0
    best_val_ep = 0
    len_train = len(train_loader)
    len_ssl = len(ssl_loader)
    len_val = len(val_loader)
    def change_model(model, freeze = True):
        for params in model.parameters():
            params.requires_grad = not freeze
    
    print("Training Started, Approx examples - Train %d SSL %d Val %d"%(len_train * bs,len_ssl * bs,len_val * bs))
    for it in range(num_iter):
        if it % len_train == 0:
            print("Iteration: ", it)
            train_iter = iter(train_loader)
            idx_train = 0
            A_train_bank =  torch.sigmoid(A_train_bank * 2)
            #print(A_train_bank)
            train_loader.dataset.weight_sample = A_train_bank
            if not it == 0:
                train_loss, train_acc = get_acc(H,W, criterion, train_loader)
                print("Train Loss: ", train_loss, "Train Acc: ", train_acc)

        if it % len_ssl == 0:
            ssl_iter = iter(ssl_loader)
            idx_ssl = 0
            A_ssl_bank =  torch.sigmoid(A_ssl_bank * 2)
            ssl_loader.dataset.weight_sample = A_ssl_bank

        if it % len_val == 0:
            val_iter = iter(val_loader)

        item_train = next(train_iter)
        item_ssl = next(ssl_iter)
        item_val = next(val_iter)

        #Update weights H and A_2 with repect to L_train and L_val respectively
        img_train, targets_train, A_train = item_train[0].cuda(), item_train[1].cuda(), item_train[2].cuda()
        A_train.requires_grad = True
        input_search, target_search = item_val[0].cuda(), item_val[1].cuda()
        logits = H(W(img_train))
        loss_A_train = criterion(logits,targets_train.cuda(),A_train.cuda())
        change_model(W,freeze=True)
        loss_A_train.backward() # Updating model 1 time
        change_model(W,freeze=False)
        unrolled_H = deepcopy(H)
        unrolled_loss = unrolled_H._loss(W(input_search), target_search, A_train)
        unrolled_loss.backward()
        unrolled_H_grad_vector = [v.grad.data for v in unrolled_H.parameters()]
        gradients = hessian_vector_product(H, unrolled_H_grad_vector, W(img_train), targets_train, A_train)
        A_train = A_train - gradients[0]
        A_train = A_train.detach()
        A_train.requires_grad = False
        A_train_bank[idx_train * bs: (idx_train + 1) * bs] = A_train
        idx_train = idx_train + 1
        optimizer_H.step()
        optimizer_H.zero_grad()
        unrolled_H.zero_grad()

        # Update weights W and A_1 with repect to L_ssl and L_val respectively
        img_ssl, targets_ssl, A_ssl = item_ssl[0].cuda(), item_ssl[1].cuda(), item_ssl[2].cuda()
        A_ssl.requires_grad = True
        input_search, target_search = item_val[0].cuda(), item_val[1].cuda()
        logits_ssl = W(img_ssl)
        loss_A_ssl = criterion(logits_ssl, targets_ssl.cuda(), A_ssl.cuda())
        loss_A_ssl.backward() # Updating model 1 time
        unrolled_W = deepcopy(W)
        # Here target needs to be changed
        unrolled_loss = unrolled_W._loss(input_search, target_search, A_ssl)
        unrolled_loss.backward()
        unrolled_W_grad_vector = [v.grad.data for v in unrolled_W.parameters()]
        gradients = hessian_vector_product(W, unrolled_W_grad_vector, img_ssl, targets_ssl, A_ssl)
        A_ssl = A_ssl - gradients[0]
        A_ssl = A_ssl.detach()
        A_ssl.requires_grad = False
        A_ssl_bank[idx_ssl * bs: (idx_ssl + 1) * bs] = A_ssl
        idx_ssl = idx_ssl + 1
        optimizer_W.step()
        optimizer_W.zero_grad()
        unrolled_W.zero_grad()
        print("Iteration done")


def val(H,W, val_loader, criterion):
    H.eval()
    W.eval()
    val_loss, val_acc = get_acc(H,W, criterion, val_loader)
    return val_loss, val_acc

def get_acc(H, W, criterion, loader):
    H.eval()
    W.eval()
    test_loss = 0
    correct = 0
    size = 0
    num_class = 10
    output_all = np.zeros((0, num_class))
    confusion_matrix = torch.zeros(num_class, num_class)
    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            im_data = data[0].cuda()
            gt_labels  = data[1].cuda()
            weight = data[2].cuda()
            output1 = H(W(im_data))
            output_all = np.r_[output_all, output1.data.cpu().numpy()]
            size += im_data.size(0)
            pred1 = output1.data.max(1)[1]
            for t, p in zip(gt_labels.view(-1), pred1.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
            correct += pred1.eq(gt_labels.data).cpu().sum()
            test_loss += criterion(output1, gt_labels, weight) / len(loader)
    #print('\nTest set: Average loss: {:.4f}, ''Accuracy: {}/{} F1 ({:.0f}%)\n'.format(test_loss, correct, size, 100. * correct / size))
    return test_loss.data, 100. * float(correct) / size



if __name__ == "__main__":
    main()




"""
print("Saving model ...")
if val_acc > best_val_acc:
    best_val_acc = val_acc
    best_val_ep = epoch
    torch.save({'epoch': epoch, 'val_acc': val_acc, 'val_loss': val_loss, 'H': H.state_dict(), 'W': W.state_dict(), 'A_ssl': A_ssl, 'A_train': A_train}, "./model_weights/checkpoint.best.pth.tar")
"""