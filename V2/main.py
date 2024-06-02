import torchvision
import torch
from torchvision import transforms
from torchvision import transforms
from Diffusion import Diffusion

if __name__ == '__main__':

    transform = transforms.Compose([
                transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # normalize mean and std for n=3 color channels
            ])
    training_set = torchvision.datasets.CIFAR10('./data', train=True, transform=transform, download=True)
    training_loader = torch.utils.data.DataLoader(training_set, batch_size=16, shuffle=True, num_workers=1,  pin_memory=True)


    model = Diffusion(noise_steps=1000).to("cuda")
    n = torch.tensor(1)
    t = torch.tensor([1])
    model.train_model(training_loader, t, 10, 1)