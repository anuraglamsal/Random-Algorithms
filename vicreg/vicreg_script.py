import torch
import torchvision
from PIL import Image

def transforms(img):
    # accompanying notes: ** add notion link **
    random_crop = torchvision.transforms.RandomResizedCrop(size = 224, scale = (0.08, 0.1))
    random_horizontal_flip = torchvision.transforms.RandomHorizontalFlip(p = 0.5)
    color_jitter = torchvision.transforms.ColorJitter(brightness = 0.4, 
                                                  contrast = 0.4, 
                                                  saturation = 0.2, 
                                                  hue = 0.1)

    # apply color jitter with a probability of 0.8. the ColorJitter class doesn't
    # have that by default, so.. Make sure to provide the transformations in a list.
    randomApply_jitter = torchvision.transforms.RandomApply([color_jitter], p = 0.8)
    random_grayscale = torchvision.transforms.RandomGrayscale(p = 0.2)
    gauss_blur = torchvision.transforms.GaussianBlur(kernel_size = 23, sigma = 1)
    randomApply_gaussBlur = torchvision.transforms.RandomApply([gauss_blur], p = 0.5)

    # if the pixel value is greater than 150, then solarization is applied to those. 
    random_solarization = torchvision.transforms.RandomSolarize(threshold = 150, p = 0.1)
    # need to understand why this is done and how the values are chosen.
    color_normalization = torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


    augmented_image = random_crop(img) # random crop

    augmented_image = random_horizontal_flip(augmented_image) # random horizontal flip

    augmented_image = randomApply_jitter(augmented_image) # random color jitter

    augmented_image = random_grayscale(augmented_image) # random grayscale

    augmented_image = randomApply_gaussBlur(augmented_image) # random gaussian blur

    augmented_image = random_solarization(augmented_image) # random solarization

    aug_img_tensor = torchvision.transforms.functional.pil_to_tensor(augmented_image).float()

    # print(aug_img_tensor.shape)
    # print(aug_img_tensor)

    aug_img_tensor = color_normalization(aug_img_tensor) # color normalization

    return aug_img_tensor

class VICRegLoss(torch.nn.Module):

    def __init__(self, lamb, mu, nu, eps):
        super(VICRegLoss, self).__init__()
        self.lamb = lamb
        self.mu = mu
        self.nu = nu
        self.eps = eps
        self.var = torch.empty()  # we will extract this from the covariance matrix i.e. the diagonal.

    def dist(self, x_aug_1, x_aug_2):
        return torch.cdist(x_aug_1, x_aug_2, p = 1.0) # (5) of paper. p = 1.0 implies L1 norm. 
    
    def regularized_covars(self, x_aug_1, x_aug_2):
        return calc_regularized_covar(x_aug_1) + calc_regularlized_covar(x_aug_2)

    def calc_regularized_covar(self, x_aug):
        covar = torch.cov(x_aug.t()) # very nice way to write (3) of paper
        self.var = torch.diag(covar) # store the variance here before zeroing it out for the regularized covar calc below. 
        regularized_covar = torch.sum(torch.square(torch.tril(covar, diagonal = -1))) # (4) of paper
        return regularized_covar
    
    def regularized_vars(self, x_aug_1, x_aug_2):
        return calc_regularized_vars(x_aug_1) + calc_regularized_vars(x_aug_2)

    def calc_regularlized_vars(self, x_aug):
        dims = x_aug.shape[1]
        sum = 0
        for el in x_aug:
            sum = sum + max(0, self.lamb - torch.sqrt(el + self.eps)) # (1) of paper
        return sum / dims

    def forward(self, x_aug_1, x_aug_2):
        return 

class SiameseBranch(torch.nn.Module):

    def __init__(self):
        super(SiameseBranch, self).__init__()

        # encoder. Just do "SiameseBranchObject.resnet" to access the resnet. 
        self.resnet = torchvision.models.resnet50(progress = True) 
        # print(resnet.fc.out_features) # This is 1000. Need to make 2048.
        self.resnet.fc = torch.nn.Linear(self.resnet.fc.in_features, 2048)
        # print(resnet.fc.out_features) # This is 2048 now.

        # expander 
        self.expander = torch.nn.Sequential(
            torch.nn.Linear(2048, 8192),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(8192),
            torch.nn.Linear(8192, 8192),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(8192),
            torch.nn.Linear(8192, 8192)
        )

    def forward(self, img):
        # encoder compute
        encoder_out = self.resnet(img)

        # expander compute
        return self.expander(encoder_out)
