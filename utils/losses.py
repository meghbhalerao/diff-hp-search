import torch 
import torch.nn as nn

class CE_Mask(nn.Module):
    def __init__(self, reduction = 'none'):
        super(CE_Mask,self).__init__()
        self.reduction = reduction
        self.criterion_ce = nn.CrossEntropyLoss(reduction = self.reduction)

    def forward(self, input, target, weight):
        if weight is not None:
            ce_loss = torch.mean(weight * self.criterion_ce(input,target))
            return ce_loss
        else:
            return torch.mean(self.criterion_ce(input,target))