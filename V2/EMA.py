import torch

# https://arxiv.org/pdf/2312.02696
class EMA:
    def __init__(self, beta=0.999):
        self.beta = beta
        self.step = 0

    def update_model(self, ema_model, model):
        with torch.no_grad():
            for current_parameters, ema_parameters in zip(model.parameters(), ema_model.parameters()):
                old_weights, new_weights = ema_parameters.data, current_parameters.data
                ema_parameters.data = self.update_average(old_weights, new_weights)

    def update_average(self, old_weights, new_weights):
        return old_weights * self.beta + (1 - self.beta) * new_weights

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())
