import os
import torch
import torchvision
import torchvision.transforms as transforms
from DDPM import DiffusionModel
from utils import *


if __name__ == "__main__":
    img_size = 64
    transform = transforms.Compose([
            transforms.Resize(80),
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    training_set = torchvision.datasets.CIFAR10('./data', train=True, transform=transform, download=True)
    testing_set = torchvision.datasets.CIFAR10('./data', train=False, transform=transform, download=True)
    training_set, validation_set = torch.utils.data.random_split(training_set, [45000, 5000])

    training_loader = torch.utils.data.DataLoader(training_set, batch_size=16, shuffle=True, num_workers=4,  pin_memory=True)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
    testing_loader = torch.utils.data.DataLoader(testing_set, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)

    model = DiffusionModel(learning_rate=0.001)

    epoch = 10

    model.train(training_loader, epoch)


    x = model.sample(16,None)
    save_images(x, os.path.join("output_images", "grid_image.png"), nrow=4, padding=2)

