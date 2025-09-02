import torch
from torchvision import transforms

augmentations = transforms.Compose([
    # RandomResizedCrop is nice because it gives you control over how much area will the crop generally cover, 
    # what will be the shape of the crop generally, and the region that is cropped itself is random too, like in 
    # the normal RandomCrop. 

    # The scale is basically Area_of_the_crop / Total_area_of_the_image. And the aspect ratio is width / height. 
    # You can see how the scale and the aspect ratio, respectively, help you control the area and shape of your crop. 
    # And if you think about it, you can solve the scale and ratio equations to find out the width and the height of the 
    # crop. And once you know that, you can randomly select a region with those width and height. 
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

# Maybe just use imagenet 1k (ILSVRC) or try looking for something else.. I think good vision learning requires
# Continual Learning though. Like, humans are seeing things all the time. Learning vision related stuff, it seems,
# is done passively all the time. Or maybe I am wrong, and ImageNet is all you need, or whatever. 
