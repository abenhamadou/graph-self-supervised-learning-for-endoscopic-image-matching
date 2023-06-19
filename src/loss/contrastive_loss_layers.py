import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np


class ContrastiveLoss(torch.nn.Module):
    def __init__(self,minibatch_size, temperature_parameter):
        torch.nn.Module.__init__(self)
        self.batch_size=minibatch_size
        self.temperature_parameter=temperature_parameter
        self.fc1 = torch.nn.Linear(128,128)
        self.fc2 = torch.nn.Linear(128, 128)

    def projection(self, z: torch.Tensor) -> torch.Tensor:

            z = F.elu(self.fc1(z))

            return self.fc2(z)


    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())


    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.temperature_parameter)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        return -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))


    def batched_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor,
                          batch_size: int):
        # Space complexity: O(BN) (semi_loss: O(N^2))
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.temperature_parameter)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))  # [B, N]
            between_sim = f(self.sim(z1[mask], z2))  # [B, N]

            losses.append(-torch.log(
                between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                / (refl_sim.sum(1) + between_sim.sum(1)
                   - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))

        return torch.cat(losses)


    def loss(self, z1: torch.Tensor, z2: torch.Tensor,
                 mean: bool = True, batch_size: int = 0):
            h1 = self.projection(z1)
            h2 = self.projection(z2)

            if batch_size == 0:
                l1 = self.semi_loss(h1, h2)
                l2 = self.semi_loss(h2, h1)
            else:
                l1 = self.batched_semi_loss(h1, h2, batch_size)
                l2 = self.batched_semi_loss(h2, h1, batch_size)

            ret = (l1 + l2) * 0.5

            ret = ret.mean() if mean else ret.sum()

            return ret
