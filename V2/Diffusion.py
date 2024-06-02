import torch
from torch import nn
from torch.nn import functional as f
from UNET import UNET
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from tqdm import tqdm


class DiffusionModel(nn.Module):
    def __init__(self, img_size=32,
                 in_channels=3,
                 out_channels=3,
                 noise_steps=1000,
                 learning_rate=0.0001,
                 beta_start=0.00085,
                 beta_end=0.0120,
                 num_classes=10,
                 context_embd_dim=256,
                 time_embd_dim=256,
                 device="cuda"):

        super().__init__()
        self.img_size = img_size
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.device = device

        self.model = UNET(
            in_channels=in_channels,
            out_channels=out_channels,
            context_embd_dim=context_embd_dim,
            time_embd_dim=time_embd_dim
        ).to(device)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        self.MSE = nn.MSELoss()
        self.scaler = torch.cuda.amp.GradScaler()

        self.beta = self.noise_scheduler()
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def noise_scheduler(self):
        return torch.linspace(self.beta_start ** 0.5, self.beta_end ** 0.5, self.noise_steps, device=self.device,
                              dtype=torch.float32) ** 2

    def add_noise(self, x_0, noise, t):
        alpha_hat_t = self.alpha_hat[t][:, None, None, None]
        x_t = torch.sqrt(alpha_hat_t) * x_0 + torch.sqrt(1.0 - alpha_hat_t) * noise
        return x_t

    def predict(self, images, context, t, noise):
        noisy_image = self.add_noise(images, noise, t)
        pred_noise = self.model(noisy_image, t, context)
        return pred_noise

    def train_model(self, training_loader, validation_loader, epochs, cfg_scale=7.5):
        self.model.train()
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=len(training_loader))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=len(training_loader), eta_min=0.00005)
        for e in range(epochs):
            running_loss = 0
            for images, labels in tqdm(training_loader):
                self.optimizer.zero_grad()
                images = images.to("cuda")
                labels = labels.to("cuda")
                time_steps = torch.randint(1, self.noise_steps, (images.shape[0],), device=self.device)
                noise = torch.randn_like(images, device=self.device)

                # for CFG
                if torch.rand(1).item() < 0.1:
                    labels = None

                with autocast():
                    pred_noise = self.predict(images, labels, time_steps, noise)
                    loss = self.MSE(noise, pred_noise)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                running_loss += loss.item()
                scheduler.step()

            print(f"Epoch: {e}: {running_loss/len(training_loader)}")


    def sample(self, n, context, cfg_scale=7.5):
        self.model.eval()
        x = torch.randn(n, 3, self.img_size, self.img_size).to(self.device)

        with torch.no_grad():
            for i in reversed(range(1, self.noise_steps)):
                if i % 100 == 0:
                    torch.cuda.empty_cache()

                t = (torch.ones(n) * i).long().to(self.device)

                # classifier free guidance
                conditional_pred = self.model(x, t, context)
                if cfg_scale > 0:
                    unconditional_pred = self.model(x, t, None)
                    conditional_pred = cfg_scale * (conditional_pred - unconditional_pred) + unconditional_pred

                # broadcasting to (noise_step, 1, 1, 1)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]

                # no noise added last step
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)

                x = 1 / torch.sqrt(alpha) * (
                            x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * conditional_pred) + torch.sqrt(
                    beta) * noise

                print(i)

            x = (x.clamp(-1, 1) + 1) / 2
            x = (x * 255).type(torch.uint8)

        return x
