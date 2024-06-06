import copy
import os
import torch
import wandb
from torch import nn
from torch.nn import functional as f
from UNET3 import UNet
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from tqdm import tqdm
from math import pi
from EMA import EMA


class DiffusionModel(nn.Module):
    def __init__(self,
                 block_structure,
                 block_multiplier,
                 d_model=64,
                 img_size=32,
                 in_channels=3,
                 out_channels=3,
                 noise_steps=1000,
                 learning_rate=0.0001,
                 min_learning_rate=0.00005,
                 beta_start=0.00085,
                 beta_end=0.0120,
                 num_classes=10,
                 context_embd_dim=256,
                 time_embd_dim=257,
                 noise_schedule="quad",
                 warm_up=1000,
                 sampler="DDPM",
                 device="cuda"):

        super().__init__()
        self.img_size = img_size
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.device = device
        self.learning_rate = learning_rate
        self.min_learning_rate = min_learning_rate
        self.warm_up = warm_up

        self.model = UNet(in_channels=3, out_channels=3, T=noise_steps, block_structure=block_structure,
                          block_multiplier=block_multiplier, d_model=d_model).to(device)

        self.EMA = EMA()
        self.EMA_model = copy.deepcopy(self.model).eval().requires_grad_(False)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.MSE = nn.MSELoss()
        self.scaler = torch.cuda.amp.GradScaler()

        self.beta = self.beta_schedule(type=noise_schedule)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def beta_schedule(self, type="quad", s=0.008):
        if type == "linear":
            return torch.linspace(self.beta_start, self.beta_end, self.noise_steps, dtype=torch.float32,
                                  device=self.device)
        elif type == "quad":
            return torch.linspace(self.beta_start ** 0.5, self.beta_end ** 0.5, self.noise_steps, device=self.device,
                                  dtype=torch.float32) ** 2
        else:
            timesteps = (torch.arange(self.noise_steps + 1, dtype=torch.float32,
                                      device=self.device) / self.noise_steps + s)
            alphas = timesteps / (1 + s) * pi / 2
            alphas = torch.cos(alphas).pow(2)
            alphas = alphas / alphas[0]
            betas = 1 - alphas[1:] / alphas[:-1]
            return betas.clamp(max=0.999)

    def warmup_lr(self, step):
        return min(step, self.warm_up) / self.warm_up

    def add_noise(self, x_0, noise, t):
        alpha_hat_t = self.alpha_hat[t][:, None, None, None]
        x_t = torch.sqrt(alpha_hat_t) * x_0 + torch.sqrt(1.0 - alpha_hat_t) * noise
        return x_t

    def training_step(self, loss):
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        # scheduler.step()

    def predict(self, images, context, model, learning=True):
        images = images.to("cuda")
        context = context.to("cuda")

        t = torch.randint(1, self.noise_steps, (images.shape[0],), device=self.device)
        noise = torch.randn_like(images, device=self.device)

        # for CFG
        if learning:
            if np.random.rand() < 0.1:
                context = None

        with autocast(dtype=torch.float16):
            noisy_image = self.add_noise(images, noise, t)
            pred_noise = model(noisy_image, t, context)
            loss = self.MSE(noise, pred_noise)

        if learning:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

        l = loss.item()
        return l

    def train_model(self, training_loader, validation_loader, epochs, log=False, save_path="save.pt"):

        #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=len(training_loader))
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=len(training_loader)*5, eta_min=self.min_learning_rate)
        self.warm_up = len(training_loader) * 4
        scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.warmup_lr)

        for e in range(epochs):
            epoch_training_loss = 0
            epoch_validation_loss = 0
            print(f"Epoch {e}...")
            self.model.train()
            # Training
            for images, labels in tqdm(training_loader):
                self.optimizer.zero_grad()
                loss = self.predict(images, labels, self.model, learning=True)
                epoch_training_loss += loss
                self.EMA.step_ema(ema_model=self.EMA_model, model=self.model)
                scheduler.step()

            self.model.eval()
            # Validation
            with torch.no_grad():
                for images, labels in tqdm(validation_loader):
                    loss = self.predict(images, labels, self.EMA_model, learning=False)
                    epoch_validation_loss += loss

            epoch_validation_loss /= len(validation_loader)
            epoch_training_loss /= len(training_loader)

            print(f"Training Loss: {epoch_training_loss}, Validation Loss: {epoch_validation_loss}")
            # Epoch logging
            if log:
                pil_image = self.sample(8, torch.tensor([1], device=self.device))
                image = wandb.Image(pil_image, caption=f"class 2")
                wandb.log(
                    {"Training_Loss": epoch_training_loss, "Validation_Loss": epoch_validation_loss, "Sample": image})
                torch.save(self.model.state_dict(), save_path)
                print("Model Saved.")

    def sample(self, n, context, cfg_scale=7.5, sampler="DDPM"):
        self.model.eval()
        x = torch.randn(n, 3, self.img_size, self.img_size).to(self.device)

        steps = self.noise_steps
        if sampler == "DDIM":
            steps -= 1

        with torch.no_grad():
            for i in reversed(range(1, steps)):

                t = (torch.ones(n) * i).long().to(self.device)

                # classifier free guidance
                conditional_pred = self.EMA_model(x, t, context)
                if cfg_scale > 0:
                    unconditional_pred = self.model(x, t, None)
                    conditional_pred = cfg_scale * (conditional_pred - unconditional_pred) + unconditional_pred

                if sampler is "DDIM":
                    x = self.DDIM_Sampler(x, conditional_pred, i, t)
                else:
                    x = self.DDPM_Sampler(x, conditional_pred, i, t)

        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)

        return x

    def load_model(self):
        self.model.load_state_dict(torch.load('model_save1_XL.pt'))

    def print_parameter_count(self):
        print(sum(p.numel() for p in self.model.parameters()))

    def DDPM_Sampler(self, x, pred, i, t):

        # broadcasting to (noise_step, 1, 1, 1)
        alpha = self.alpha[t][:, None, None, None]
        alpha_hat = self.alpha_hat[t][:, None, None, None]
        beta = self.beta[t][:, None, None, None]

        # no noise added last step
        if i > 1:
            noise = torch.randn_like(x)
        else:
            noise = torch.zeros_like(x)

        x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * pred) + torch.sqrt(beta) * noise
        return x

    def DDIM_Sampler(self, x, pred, i, t, eta=0):

        # broadcasting to (noise_step, 1, 1, 1)

        alpha_t_prev = self.alpha[t + 1][:, None, None, None]
        alpha_t = self.alpha[t][:, None, None, None]

        # no noise added last step
        if i > 1:
            noise = torch.randn_like(x)
        else:
            noise = torch.zeros_like(x)



        sigma_t = eta * torch.sqrt((1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev))
        epsilon_t = torch.randn_like(x)
        x_t_minus_one = (torch.sqrt(alpha_t_prev / alpha_t) * x + (torch.sqrt(1 - alpha_t_prev - sigma_t ** 2) -
                         torch.sqrt((alpha_t_prev * (1 - alpha_t)) / alpha_t)) * pred + sigma_t * epsilon_t
        )

        return x
