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

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True, num_workers=2, drop_last = True)

    val_loader = torch.utils.data.DataLoader(val_data, batch_size=128, shuffle=False, num_workers=2, drop_last = True)

    criterion = CE_Mask().cuda()
    model = AlexNet(criterion = CE_Mask, num_classes = 10).cuda()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    num_epochs = 100
    learning_rate_min =  0.001
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(num_epochs), eta_min=learning_rate_min)
    weight_bank = weight_sample
    best_val_acc = 0
    best_val_ep = 0
    for epoch in range(num_epochs):
        print("Epoch #: ", epoch)
        train_loss, train_acc = train(model, train_loader, val_loader, criterion, weight_bank, optimizer)
        train_loss = train_loss.cpu().data.item()
        print("Train Acc: ", train_acc, "Train Loss: ", train_loss)
        val_loss, val_acc = val(model, val_loader, criterion)
        val_loss = val_loss.cpu().data.item()
        print("Val Acc: ", val_acc,"Val Loss: ", val_loss)
        # Initialize the train_loader here again
        print("Saving model ...")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_ep = epoch
            torch.save({'epoch': epoch, 'val_acc': val_acc, 'val_loss': val_loss, 'model': model.state_dict()}, "./model_weights/model_best.ckpt.best.pth.tar")
            torch.save(weight_bank,"./model_weights/weight_bank.pth")
        print("Best Val Acc: ", best_val_acc)
        scheduler.step()

def train(model, train_loader, val_loader, criterion, weight_bank, optimizer):
    model.train()
    batch_size = train_loader.batch_size
    for idx, (img, targets, weight, _) in enumerate(train_loader):
        img, targets, weight = img.cuda(), targets.cuda(), weight.cuda()
        weight.requires_grad = True
        input_search, target_search, _ , _ = next(iter(val_loader)) # Querying the validation image and targets to update A
        input_search, target_search = input_search.cuda(), target_search.cuda()
        logits = model(img)
        loss_weights = criterion(logits,targets.cuda(),weight.cuda())
        loss_weights.backward() # Updating model 1 time
        
        # Making a separate network for a one step unrolled model
        unrolled_model = deepcopy(model)
        unrolled_loss = unrolled_model._loss(input_search, target_search, weight)
        unrolled_loss.backward()

        # Update A i.e. weights for the loss functions
        unrolled_model_grad_vector = [v.grad.data for v in unrolled_model.parameters()]
        gradients = hessian_vector_product(model, unrolled_model_grad_vector, img, targets, weight)
        weight = weight - gradients[0]
        weight = weight.detach()
        weight.requires_grad = False
        weight_bank[idx * batch_size: (idx + 1) * batch_size] = weight
        optimizer.step()
        optimizer.zero_grad()
        unrolled_model.zero_grad()
        break

    weight_bank =  torch.sigmoid(weight_bank * 2) # Squeezing the weight values between 0 and 1 to make it like a probablity 
    print(weight_bank)
    train_loader.dataset.weight_sample = weight_bank
    train_loss, train_acc = get_acc(model, criterion, train_loader)
    return train_loss, train_acc

def val(model, val_loader, criterion):
    model.eval()
    val_loss, val_acc = get_acc(model, criterion, val_loader)
    return val_loss, val_acc

def get_acc(model, criterion, loader):
    model.eval()
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
            output1 = model(im_data)
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





















#for att in dir(train_loader.dataset):
#    print(att,getattr(train_loader.dataset, att))
# Definfing the hyperparameter A's weights 
