import torch
from torch import nn
from torch.nn import functional as f
from UNET import UNET


class Diffusion(nn.Module):
    def __init__(self, img_size=32, noise_steps=1000, learning_rate=0.001, beta_start=0.00085, beta_end=0.0120, device="cuda"):
        super().__init__()
        self.img_size = img_size
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.device = device

        self.model = UNET().to(device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)

        self.beta = self.noise_scheduler()
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def noise_scheduler(self):
        return torch.linspace(self.beta_start ** 0.5, self.beta_end ** 0.5, self.noise_steps, device=self.device, dtype=torch.float32) ** 2

    def add_noise(self, x_0, noise, t):
        alpha_hat_t = self.alpha_hat[t]
        x_t = torch.sqrt(alpha_hat_t)*x_0 + torch.sqrt(1.0 - alpha_hat_t)*noise
        return x_t

    def forward(self, n, context, cfg_scale=7.5):
        self.model.eval()
        x = torch.randn(n, 3, self.img_size, self.img_size).to(self.device)

        for i in reversed(range(1, self.noise_steps)):

            t = (torch.ones(n)*i).long().to(self.device)

            # classifier free guidance
            conditional_pred = self.model(x, t, context)
            if cfg_scale > 0:
                unconditional_pred = self.model(x, t, None)
                conditional_pred = cfg_scale * (conditional_pred - unconditional_pred) + unconditional_pred

            # broadcasting to (noise_step, 1, 1, 1)
            alpha = self.alpha[t][:, None, None, None]
            alpha_hat = self.alpha_hat[t][:, None, None, None]
            beta = self.beta[t][:, None, None, None]

            if i > 1:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)

            x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * conditional_pred) + torch.sqrt(
                beta) * noise


        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x


    def predict(self, x_t, t, x_0):
        pass
    def train_model(self):
        self.model.train()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.noise_steps)



model = Diffusion(noise_steps=10).to("cuda")
n = torch.tensor(1)
t = torch.tensor([1])
model(n, t)