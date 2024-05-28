import os
import torch
import torchvision
import torchvision.transforms as transforms
from DDPM import DiffusionModel
from utils import *
import numpy as np

if __name__ == "__main__":
    img_size = 32
    transform = transforms.Compose([
            transforms.Resize(img_size + int(.25*img_size)),     # Scale up by 25% to enable random crop
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    training_set = torchvision.datasets.CIFAR10('./data', train=True, transform=transform, download=True)
    testing_set = torchvision.datasets.CIFAR10('./data', train=False, transform=transform, download=True)
    training_set, validation_set = torch.utils.data.random_split(training_set, [45000, 5000])

    training_loader = torch.utils.data.DataLoader(training_set, batch_size=10, shuffle=True, num_workers=10,  pin_memory=True)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=10, shuffle=True, num_workers=10, pin_memory=True)
    testing_loader = torch.utils.data.DataLoader(testing_set, batch_size=64, shuffle=True, num_workers=10, pin_memory=True)

    model = DiffusionModel(learning_rate=0.0003, img_size=img_size, in_channels=3, out_channels=3)


    '''sample = next(iter(training_set))[0].clone().detach()
    sample = (sample * 0.5 + 0.5) * 255  # De-normalize and scale to [0, 255]
    sample = sample.numpy().astype(np.uint8)
    sample = np.transpose(sample, (1, 2, 0))  # Change from (C, H, W) to (H, W, C)
    sample = Image.fromarray(sample)
    sample.save(os.path.join("output_images", "grid_image.png"))'''
    epoch = 200

    model.train(training_loader, epoch)




