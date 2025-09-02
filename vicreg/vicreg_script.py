import torch
import torch.nn as nn
from torchvision import models
from PIL import Image

class VICRegLoss(nn.Module):

    def __init__(self, lamb = 25, mu = 25, nu = 1):
        super(VICRegLoss, self).__init__()
        self.lamb = lamb
        self.mu = mu
        self.nu = nu
        self.var_1 = torch.empty()  # We will extract this from the covariance matrix i.e. the diagonal. This is for the first set of augs.
        self.var_2 = torch.empty()  # This is for the second set of augs. 

    def dist(self, x_aug_1, x_aug_2):
        return torch.cdist(x_aug_1, x_aug_2, p = 1.0) / x_aug_1.shape[0] # (5) of paper. p = 1.0 implies L1 norm. 
    
    def regularized_covars(self, x_aug_1, x_aug_2):
        return calc_regularized_covar(x_aug_1, 1) + calc_regularlized_covar(x_aug_2, 2)

    def calc_regularized_covar(self, x_aug, idx): # idx = to store variances of the batches separately -- in self.var_1 and self.var_2.
        covar = torch.cov(x_aug.t()) # very nice way to write (3) of paper
        if idx == 1:
            self.var_1 = torch.diag(covar) # store the variance here before zeroing it out for the regularized covar calc below. 
        else:
            self.var_2 = torch.diag(covar)
        regularized_covar = torch.sum(torch.square(torch.tril(covar, diagonal = -1))) # (4) of paper
        return regularized_covar
    
    def regularized_vars(self, x_aug_1, x_aug_2):
        return calc_regularized_vars(x_aug_1, 1) + calc_regularized_vars(x_aug_2, 2)

    def calc_regularlized_vars(self, x_aug, idx, upper_limit = 1, eps = 0.0001):
        dims = x_aug.shape[1]
        sum = 0
        for el in var_1 if idx == 1 else var_2:
            sum = sum + max(0, upper_limit - torch.sqrt(el + eps)) # (1) of paper. 
        return sum / dims

    def forward(self, x_aug_1, x_aug_2):
        weighted_dist = self.lamb * dist(x_aug_1, x_aug_2)
        weighted_covars = self.nu * regularized_covars(x_aug_1, x_aug_2)
        weighted_vars = self.mu * regularized_vars(x_aug_1, x_aug_2)
        return weighted_dist + weighted_covars + weighted_vars

class SiameseBranch(nn.Module):

    def __init__(self):
        super(SiameseBranch, self).__init__()

        # encoder. Just do "SiameseBranchObject.resnet" to access the resnet. 
        self.resnet = models.resnet50(progress = True) 
        # print(resnet.fc.out_features) # This is 1000. Need to make 2048.
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 2048)
        # print(resnet.fc.out_features) # This is 2048 now.

        # expander 
        self.expander = nn.Sequential(
            nn.Linear(2048, 8192),
            nn.ReLU(),
            nn.BatchNorm1d(8192),
            nn.Linear(8192, 8192),
            nn.ReLU(),
            nn.BatchNorm1d(8192),
            nn.Linear(8192, 8192)
        )

    def forward(self, img):
        # encoder compute
        encoder_out = self.resnet(img)

        # expander compute
        return self.expander(encoder_out)
