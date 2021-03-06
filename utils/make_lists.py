import os
import random
train_path = os.path.join("./data/cifar/train/")
test_path = os.path.join("./data/cifar/test/")
label_dict = {}
label_file = open(os.path.join("./data/cifar/labels.txt"))
i = 0
for line in label_file:
    line = line.replace("\n","")
    label_dict[line] = i
    i +=1

f_ssl = open(os.path.join("ssl.txt"),"w")
f_train = open(os.path.join("train.txt"),"w")
f_val = open(os.path.join("val.txt"),"w")
f_test = open(os.path.join("test.txt"),"w")


for img in os.listdir(train_path):
    res = ''.join([i for i in str(img) if not i.isdigit()]) 
    res = res.replace("_","")
    res = res.replace(".png","")
    #f_test.write(img + " " + str(label_dict[res]) + "\n")
    rand_num = random.random()
    if rand_num > 0.1 and rand_num<0.9:
        f_ssl.write(img + " " + str(label_dict[res]) + "\n")
    elif rand_num >= 0.9:
        f_val.write(img + " " + str(label_dict[res]) + "\n")
    elif rand_num <= 0.1:
        f_train.write(img + " " + str(label_dict[res]) + "\n")
