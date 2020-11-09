import numpy as np
import os
import os.path
from PIL import Image


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def make_dataset_fromlist(image_list):
    with open(image_list) as f:
        image_index = [x.split(' ')[0] for x in f.readlines()]
    with open(image_list) as f:
        label_list = []
        selected_list = []
        for ind, x in enumerate(f.readlines()):
            label = x.split(' ')[1].strip()
            label_list.append(int(label))
            selected_list.append(ind)
        image_index = np.array(image_index)
        label_list = np.array(label_list)
    image_index = image_index[selected_list]
    return image_index, label_list



class Imagelists(object):
    def __init__(self, image_list, weight_sample, root="./data/cifar/train/", transform=None):
        imgs, labels = make_dataset_fromlist(image_list)
        self.imgs = imgs
        self.labels = labels
        self.transform = transform
        self.loader = pil_loader
        self.root = root
        self.weight_sample = weight_sample

    #def update_weight_sample(self, new_weight_sample):
        #return new_weight_sample

    def __getitem__(self, index):
        path = os.path.join(self.root, self.imgs[index])
        target = self.labels[index]
        img = self.loader(path)
        #weight_sample = self.update_weight_sample(new_weight_sample)
        weight_sample_idx = self.weight_sample[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, target, weight_sample_idx, self.imgs[index]

    def __len__(self):
        return len(self.imgs)