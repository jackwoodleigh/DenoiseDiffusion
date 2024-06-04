import torchvision
import torch
import os
from utils import *

from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision import transforms
from Diffusion import DiffusionModel

if __name__ == '__main__':

    transform = transforms.Compose([
                transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # normalize mean and std for n=3 color channels
            ])
    training_set = torchvision.datasets.CIFAR10('./data', train=True, transform=transform, download=True)



    training_subset = Subset(training_set, list(range(5000)))
    training_loader = DataLoader(training_subset, batch_size=32, shuffle=True, num_workers=1,  pin_memory=True)


    model = DiffusionModel(learning_rate=0.001).to("cuda")
    '''n = torch.tensor(1)
    t = torch.tensor([1])
    model.train_model(training_loader, training_loader, 100, log=False)'''

    model.load_model()
    c = torch.tensor([1 for i in range(16)], device="cuda")

    x = model.sample(16, c)
    save_images(x, os.path.join("output_images", "grid_image.png"), nrow=4, padding=2)