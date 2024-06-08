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
    training_loader = DataLoader(training_set, batch_size=64, shuffle=True, num_workers=4,  pin_memory=True)

    img_size = 32
    batch_size = 64
    epochs = 200
    learning_rate = 0.0001
    min_learning_rate = 0.00001
    log = True
    noise_steps = 1000
    noise_schedule = "linear"
    block_structure = [1, 1, 1]
    block_multiplier = [2, 2, 2]
    d_model = 64
    ema_weight = 0.9995

    model = DiffusionModel(
        d_model=d_model,
        block_structure=block_structure,
        block_multiplier=block_multiplier,
        learning_rate=learning_rate,
        min_learning_rate=min_learning_rate,
        in_channels=3,
        out_channels=3,
        noise_steps=noise_steps,
        beta_start=1e-4,
        beta_end=0.02,
        img_size=img_size,
        noise_schedule=noise_schedule,
        ema_weight=ema_weight,
        device="cuda"
    )
    ''' n = torch.tensor(1)
    t = torch.tensor([1])
    model.train_model(training_loader, training_loader, 100, log=False)'''

    model.load_model()
    c = torch.tensor([1 for i in range(16)], device="cuda")

    x = model.sample(16, c)
    save_images(x, os.path.join("output_images", "grid_image.png"), nrow=4, padding=2)