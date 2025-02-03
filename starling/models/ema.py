import torch
import torch.nn as nn


class EMA:
    def __init__(self, beta: float = 0.999):
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(
            current_model.parameters(), ma_model.parameters()
        ):
            ema_model_weights, current_weighs = ma_params.data, current_params.data
            ma_params.data = self.update_average(ema_model_weights, current_weighs)

    def update_average(self, ema_model_params, current_model_params):
        if ema_model_params is None:
            return current_model_params

        return ema_model_params * self.beta + (1 - self.beta) * current_model_params

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())
