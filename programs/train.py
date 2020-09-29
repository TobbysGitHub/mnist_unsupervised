import os
import torch
from matplotlib.axes import Axes
from torch.utils.data import TensorDataset, DataLoader, Dataset
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
from tqdm import tqdm
from torchvision.transforms import transforms

import math

from programs import Model
from programs.models import my_transforms

CTR = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class State:
    def __init__(self):
        self.model = None
        self.epoch = 0
        self.batch = 0
        self.steps = 0
        self.writer = SummaryWriter()

    @staticmethod
    def log(s):
        # sample log
        return s % int(np.sqrt(s) + 1) == 0

    def add_scalar(self, tag, scalar_value):
        if not self.log(self.steps):
            return
        try:
            self.writer.add_scalar(tag=tag, scalar_value=scalar_value, global_step=self.steps)
        except ValueError:
            pass

    def add_histogram(self, tag, values):
        if not self.log(self.steps):
            return
        try:
            self.writer.add_histogram(tag=tag, values=values, global_step=self.steps)
        except ValueError:
            pass


def grad_rise(model):
    imgs = torch.rand(size=(model.num_units, model.size, model.size)).to(device)
    imgs.requires_grad_(True)

    optim = torch.optim.SGD(params=[imgs],
                            momentum=0,
                            lr=0.2,
                            weight_decay=0.1)

    for i in range(100):
        y = model.encode(imgs)
        target = -y.trace()
        optim.zero_grad()
        target.backward()
        optim.step()

    imgs = imgs.detach().cpu().numpy()
    return imgs


def visualize(model):
    imgs = grad_rise(model)

    fig, a = plt.subplots(math.ceil(model.num_units / 8), 8, figsize=(4, 4))
    for i, img in enumerate(imgs):
        axis: Axes = a[i // 8][i % 8]
        axis.get_xaxis().set_visible(False)
        axis.get_yaxis().set_visible(False)
        axis.imshow(img, cmap='Gray')
    plt.show()
    return imgs


def prepare_data_loader(data_dir, batch_size):
    class MnistDataSet(MNIST):
        def __init__(self, root):
            super().__init__(root, download=True)
            self.data, self.targets = self.data.to(device), self.targets.to(device)
            self.my_transform = transforms.Compose(
                [transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5),
                 transforms.ToTensor(),
                 transforms.Normalize(0, 1),
                 my_transforms.AddGaussianNoise(0, 0.1)]
            )

        def __getitem__(self, item):
            img, target = super().__getitem__(item)
            img0 = self.my_transform(img)
            img1 = self.my_transform(img)
            return img0, img1, target

    data = MnistDataSet(data_dir)

    data_loader = DataLoader(dataset=data,
                             batch_size=batch_size,
                             shuffle=True,
                             drop_last=True)
    return data_loader


def adjust_learning_rate(optimizers, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 180 epochs"""
    epoch = state.epoch
    lr = lr * (0.1 ** (epoch // 180))
    for optimizer in optimizers:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def cal_loss(y1, y2, w, mask):
    """
    :param y1:  s_b * n_u
    :param y2:  s_b * n_u
    :param w:   s_b(q) * s_b * num_units
    :param mask: s_b * num_units
    """

    l_1_2 = torch.exp(-torch.abs(y1 - y2).clamp_max(5))  # s_b * n_u
    l_1_neg = torch.sum(w * torch.exp(-torch.abs(y1.unsqueeze(1) - y1).clamp_max(5)), dim=1)  # s_b * n_u
    l_2_neg = torch.sum(w * torch.exp(-torch.abs(y2.unsqueeze(1) - y1).clamp_max(5)), dim=1)

    loss = -torch.log(l_1_2 / l_1_neg) - torch.log(l_1_2 / l_2_neg)
    loss_all = loss.mean()
    loss = loss.masked_fill(~mask, 0)
    loss = loss.mean()

    state.add_histogram(tag='y1', values=y1)
    state.add_histogram(tag='l12', values=l_1_2)
    state.add_histogram(tag='l1neg', values=l_1_neg)
    state.add_histogram(tag='l2neg', values=l_2_neg)
    state.add_scalar(tag='loss', scalar_value=loss_all.item())

    l_1_neg_ctr = torch.mean(torch.exp(-torch.abs(y1.unsqueeze(1) - y1).clamp_max(5)), dim=1)  # s_b * n_u
    l_2_neg_ctr = torch.mean(torch.exp(-torch.abs(y2.unsqueeze(1) - y1).clamp_max(5)), dim=1)

    loss_ctr = -torch.log(l_1_2 / l_1_neg_ctr) - torch.log(l_1_2 / l_2_neg_ctr)
    loss_all_ctr = loss_ctr.mean()
    loss_ctr = loss_ctr.masked_fill(~mask, 0)
    loss_ctr = loss_ctr.mean()

    state.add_scalar(tag='loss_', scalar_value=(loss_all_ctr - loss_all).item())

    return loss_ctr if CTR else loss


def flip_grad(optimizer):
    for group in optimizer.param_groups:
        for p in group['params']:
            if p.grad is None:
                continue
            p.grad.data = - p.grad.data


def optimize(loss, optimizers):
    for optim in optimizers:
        optim.zero_grad()
    loss.backward()
    optim = optimizers[0]
    optim.step()
    optim = optimizers[1]
    flip_grad(optim)
    optim.step()


def train_epoch(model, data_loader, optimizers, ):
    for batch in data_loader:
        (y1, w), y2, mask = model(batch[0], batch[1])
        loss = cal_loss(y1, y2, w, mask)
        optimize(loss, optimizers)
        state.steps += 1
    pass


def train(data_dir='../data',
          model_config={'size': 28},
          training_config={'epochs': 200, 'lr': 0.1, 'batch_size': 128}):
    global state
    state = State()
    model: Model = Model(**model_config)
    model = model.to(device)
    state.model = model
    batch_size = training_config['batch_size']
    data_loader = prepare_data_loader(data_dir, batch_size)
    lr = training_config['lr']
    optimizers = [torch.optim.SGD(params=model.encoder.parameters(),
                                  lr=lr, momentum=0.9, weight_decay=0.001),
                  torch.optim.SGD(params=model.predictor.parameters(),
                                  lr=lr, momentum=0.9, weight_decay=0.001), ]

    epochs = training_config['epochs']
    for epoch in tqdm(range(epochs)):
        state.epoch = epoch
        adjust_learning_rate(optimizers, lr)
        train_epoch(model, data_loader, optimizers, )
        # visualize(model, )
        torch.save(model.state_dict(), f=os.path.join('state_dicts', 'model.state_dict.' + str(epoch)))


if __name__ == '__main__':
    train()
