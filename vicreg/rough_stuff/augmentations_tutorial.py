import torch
import torchvision
from PIL import Image

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


augmented_image = random_crop(Image.open("trees.jpg")) # random crop

augmented_image = random_horizontal_flip(augmented_image) # random horizontal flip

augmented_image = randomApply_jitter(augmented_image) # random color jitter

augmented_image = random_grayscale(augmented_image) # random grayscale

augmented_image = randomApply_gaussBlur(augmented_image) # random gaussian blur

augmented_image = random_solarization(augmented_image) # random solarization

# aug_img_tensor = torchvision.transforms.functional.pil_to_tensor(augmented_image).float()

# print(aug_img_tensor.shape)
# print(aug_img_tensor)

aug_img_tensor = color_normalization(aug_img_tensor)

transform_to_pil = torchvision.transforms.ToPILImage()
transform_to_pil(aug_img_tensor).show()

# print(aug_img_tensor.shape)
# print(aug_img_tensor)
# augmented_image.show()
