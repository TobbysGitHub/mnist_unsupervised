import torch
import torch.nn as nn

from .encoder import Encoder
from .predictor import Predictor


class Model(nn.Module):
    def __init__(self, num_units=64, size=16, mask_p=0.2):
        super().__init__()
        self.num_units = num_units
        self.size = size
        self.encoder = Encoder(num_units=num_units, dim_inputs=size * size, dim_hidden=32)
        self.predictor = Predictor(num_units=num_units, dim_hidden=num_units * 4, mask_p=mask_p)

    # @staticmethod
    # def gen_noise(x):
    #     noise = 0.1 * torch.randn_like(x)
    #     return noise

    def forward(self, x1, x2):
        # x = x.view(-1, self.size * self.size)
        x1, x2 = x1.view(-1, self.size * self.size), x2.view(-1, self.size * self.size)

        # x1 = x + self.gen_noise(x)
        y1 = self.encoder(x1)  # s_b * n_u
        w, mask = self.predictor(y1)  # s_b(q) * s_b * num_units

        # x2 = x + self.gen_noise(x)
        y2 = self.encoder(x2)

        return (y1, w), y2, mask

    def encode(self, x):
        x = x.view(-1, self.size * self.size)
        y = self.encoder(x)

        return y
