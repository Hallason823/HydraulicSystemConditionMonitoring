import torch
from torch import nn

class ContrastiveLoss(nn.Module):
    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = torch.dist(output1, output2)
        loss_contrastive = torch.mean((1-label)*euclidean_distance+(label)*max(0, self.margin - euclidean_distance))
        return loss_contrastive