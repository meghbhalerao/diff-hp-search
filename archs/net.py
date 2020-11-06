import torch.nn as nn
from torchvision import models


class AlexNetBase(nn.Module):
    def __init__(self, num_classes = 10,  pret=True):
        super(AlexNetBase, self).__init__()
        self.num_classes = num_classes
        model_alexnet = models.alexnet(pretrained=pret)
        self.features = nn.Sequential(*list(model_alexnet.
                                            features._modules.values())[:])
        self.classifier = nn.Sequential()
        for i in range(6):
            self.classifier.add_module("classifier" + str(i),
                                       model_alexnet.classifier[i])
        self.__in_features = model_alexnet.classifier[6].in_features
        self.last_classifier = nn.Linear(4096,self.num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        x = self.last_classifier(x)
        return x

    def output_num(self):
        return self.__in_features