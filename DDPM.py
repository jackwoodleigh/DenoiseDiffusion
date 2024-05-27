
import torch
import numpy as np
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm
from UNet import UNet


class DiffusionModel:
    def __init__(self,
                 learning_rate=0.001,
                 in_channels=3,
                 out_channels=3,
                 time_dim=256,
                 noise_steps=100,
                 beta_start=1e-4,
                 beta_end=0.02,
                 img_size=64,
                 device="cuda",
                 num_class=10
                 ):

        self.learning_rate = learning_rate
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device
        self.num_class = num_class

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.model = UNet(in_channels=in_channels, out_channels=out_channels, time_dim=time_dim, num_classes=num_class).to(device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        self.MSE = nn.MSELoss()
        self.scaler = torch.cuda.amp.GradScaler()

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_min_alpha_hat = torch.sqrt(1. - self.alpha_hat[t])[:, None, None, None]
        e = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_min_alpha_hat * e, e

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))


    def sample(self, n, labels, cfg_scale=3):
        self.model.eval()
        with torch.no_grad():

            # selecting random noise to begin diffusion process
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)

            # looping for the number of time steps
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):

                # creating tensor of time step values
                t = (torch.ones(n) * i).long().to(self.device)

                # predicting the noise using the labels.
                pred_noise = self.model(x, t, labels)

                # If using classifier free guidance
                if cfg_scale > 0:
                    unconditional_pred_noise = self.model(x, t, None)
                    pred_noise = torch.lerp(unconditional_pred_noise, pred_noise, cfg_scale)

                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]


                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)

                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * pred_noise) + torch.sqrt(beta) * noise

            self.model.train()

            x = (x.clamp(-1, 1) + 1) / 2
            x = (x * 255).type(torch.uint8)
            return x

    def train(self, train_loader, epochs):
        for e in range(epochs):
            running_loss = 0
            pbar = tqdm(train_loader)
            for i, (images, labels) in enumerate(pbar):
                images = images.to(self.device)
                labels = labels.to(self.device)
                self.optimizer.zero_grad()

                # sample some random time steps
                t = self.sample_timesteps(images.shape[0]).to(self.device)

                # create noisy images at random time steps
                x_t, noise = self.noise_images(images, t)

                if np.random.random() < 0.1:
                    labels = None

                pred_noise = self.model(x_t, t, labels)
                loss = self.MSE(noise, pred_noise)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                running_loss += loss.item()

            print(running_loss/len(train_loader))

