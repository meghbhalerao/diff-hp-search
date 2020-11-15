import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 2 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 2 * 2)
        x = self.classifier(x)
        return x

    def _loss(self, input, target, mask, H = None):
        if H is not None:
            logits = H(self(input))
        else:
            logits = self(input)
        logits = self(input)
        return self._criterion(logits, target, mask)    
    
class Predictor(nn.Module):
    def __init__(self, num_class=10, inc=4096, temp=0.05):
        super(Predictor, self).__init__()
        self.fc = nn.Linear(inc, num_class, bias=False)
        self.num_class = num_class
        self.temp = temp

    def forward(self, x, reverse=False, eta=0.1):
        x_out = self.fc(x) / self.temp
        return x_out

    def _loss(self, input, target, mask):
        logits = self(input)
        return self._criterion(logits, target, mask) 
    
    
 
class Predictor_deep(nn.Module):
    def __init__(self, criterion, num_class=10, inc=4096, temp=0.05):
        super(Predictor_deep, self).__init__()
        self.fc1 = nn.Linear(inc, 512)
        self.fc2 = nn.Linear(512, num_class, bias=False)
        self.num_class = num_class
        self.temp = temp
        self._criterion = criterion()

    def forward(self, x, reverse=False, eta=0.1):
        x = self.fc1(x)
        x = F.normalize(x)
        x_out = self.fc2(x) / self.temp
        return x_out

    def _loss(self, input, target, mask):
        logits = self(input)
        return self._criterion(logits, target, mask) 
    
