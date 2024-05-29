import os
import torch
import torchvision
import torchvision.transforms as transforms
from DDPM import DiffusionModel
from utils import *
import numpy as np

if __name__ == "__main__":
    img_size = 32
    batch_size = 10

    transform = transforms.Compose([
        transforms.Resize(img_size + int(.25 * img_size)),  # Scale up by 25% to enable random crop
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # normalize mean and std for n=3 color channels
    ])

    training_set = torchvision.datasets.CIFAR10('./data', train=True, transform=transform, download=True)
    testing_set = torchvision.datasets.CIFAR10('./data', train=False, transform=transform, download=True)
    training_set, validation_set = torch.utils.data.random_split(training_set, [45000, 5000])

    training_loader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=10,
                                                  pin_memory=True)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=True, num_workers=10,
                                                    pin_memory=True)
    testing_loader = torch.utils.data.DataLoader(testing_set, batch_size=batch_size, shuffle=True, num_workers=10,
                                                 pin_memory=True)

    model = DiffusionModel(
        learning_rate=0.001,
        in_channels=3,
        out_channels=3,
        noise_steps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        img_size=img_size,
        device="cuda",
        num_class=10
    )

    epochs = 200
    model.train(training_loader, validation_loader, epochs)



