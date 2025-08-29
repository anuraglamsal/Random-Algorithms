import torch
from torchvision import transforms

augmentations = transforms.Compose([
    transforms.RandomResizedCrop(size = 224, scale = (0.08, 0.1)),
    transforms.RandomHorizontalFlip(p = 0.5),
    # apply color jitter with a probability of 0.8. the ColorJitter class doesn't
    # have that by default, so.. Make sure to provide the transformations in a list.
    transforms.RandomApply([transforms.ColorJitter(brightness = 0.4, 
                                                   contrast = 0.4, 
                                                   saturation = 0.2, 
                                                   hue = 0.1)], 
                           p = 0.8),
    transforms.RandomGrayscale(p = 0.2),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size = 23, sigma = 1)
                            ], p = 0.5),
    # if the pixel value is greater than 150, then solarization is applied to those. 
    transforms.RandomSolarize(threshold = 150, p = 0.1),
    # need to understand why this is done and how the values are chosen.
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])
