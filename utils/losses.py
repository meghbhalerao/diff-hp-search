import torch 
import torch.nn as nn

class CE_Mask(nn.Module):
    def __init__(self, reduction = 'none'):
        super(CE_Mask,self).__init__()
        self.reduction = reduction
        self.criterion = nn.CrossEntropyLoss(reduction = self.reduction)

    def forward(self, input, target, mask):
        ce_loss = torch.mean(mask * self.criterion(input,target))
        return ce_loss