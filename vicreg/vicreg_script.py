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

    def __init__(self, lamb, mu, nu):
        super(VICRegLoss, self).__init__()
        self.lamb = lamb
        self.mu = mu
        self.nu = nu

    def dist(self, x_aug_1, x_aug_2):
        return torch.cdist(x_aug_1, x_aug_2, p = 1.0)

    def var(self):
        return

    def covar(self):
        return

    def forward(self, x_aug_1, x_aug_2):
        return 
"""
resnet = torchvision.models.resnet50(progress = True)
# print(resnet.fc.out_features) # This is 1000. Need to make 2048.
resnet.fc = torch.nn.Linear(resnet.fc.in_features, 2048)
print(resnet.fc.out_features) # This is 2048 now. 
"""

class SiameseBranch(torch.nn.Module):

    def __init__(self):
        super(SiameseBranch, self).__init__()

        # encoder
        self.resnet = torchvision.models.resnet50(progress = True)
        # print(resnet.fc.out_features) # This is 1000. Need to make 2048.
        self.resnet.fc = torch.nn.Linear(self.resnet.fc.in_features, 2048)
        # print(resnet.fc.out_features) # This is 2048 now.

        # expander 
        self.fcl1 = torch.nn.Linear(2048, 8192)
        self.bn1 = torch.nn.BatchNorm1d(8192)
        self.fcl2 = torch.nn.Linear(8192, 8192)
        self.bn2 = torch.nn.BatchNorm1d(8192)
        self.linear1 = torch.nn.Linear(8192, 8192)
        self.relu = torch.nn.ReLU()

    def forward(self, img):
        # encoder compute
        encoder_out = self.resnet(img)

        # expander compute
        first_layer_out = self.bn1(self.relu(self.fcl1(encoder_out)))
        second_layer_out = self.bn2(self.relu(self.fcl2(first_layer_out)))
        final_out = self.linear1(second_layer_out)
        return final_out

"""
class test:
    def met1(self):
        print("test")
    def met2(self):
        self.met1()
        
x = test()
x.met2()
"""
