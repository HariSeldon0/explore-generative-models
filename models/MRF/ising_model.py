import random
import numpy as np
import torch


class Ising:
    def __init__(self, n, J, x=None):
        self.J = J
        self.n = n
        self.x = x
        if self.x is None:
            self.x = self.random_x()

    def random_x(self):
        return torch.randint(-1, 1, (self.n, self.n)) * 2 + 1

    def unnormalized_p(self, x=None):
        """input: x
           output: unnormalizd probability of x

        Args:
            x (Tensor): size = (n * n)
        """
        if x is None:
            x = self.x

        x_left = torch.roll(x, -1, 1)
        x_up = torch.roll(x, -1, 0)
        E = -self.J * (torch.sum((x * x_left)[:, :-1]) + torch.sum((x * x_up)[:-1, :]))
        return torch.exp(-E)

    def _stable_prob(self, x, pos):
        x[pos[0], pos[1]] = -1
        x_left = torch.roll(x, -1, 1)
        x_up = torch.roll(x, -1, 0)
        E0 = -self.J * (torch.sum((x * x_left)[:, :-1]) + torch.sum((x * x_up)[:-1, :]))

        x[pos[0], pos[1]] = 1
        x_left = torch.roll(x, -1, 1)
        x_up = torch.roll(x, -1, 0)
        E1 = -self.J * (torch.sum((x * x_left)[:, :-1]) + torch.sum((x * x_up)[:-1, :]))

        return 1.0 / (1.0 + torch.exp(E1 - E0))

    def _stable_effient_prob(self, x, pos):
        neighbors = []
        if pos[0] - 1 >= 0:
            neighbors.append(x[pos[0] - 1, pos[1]])
        if pos[0] + 1 < self.n:
            neighbors.append(x[pos[0] + 1, pos[1]])
        if pos[1] - 1 >= 0:
            neighbors.append(x[pos[0], pos[1] - 1])
        if pos[1] + 1 < self.n:
            neighbors.append(x[pos[0], pos[1] + 1])
        neighbors = torch.tensor(neighbors, dtype=torch.float)

        E0 = -self.J * (torch.sum(-1 * neighbors))
        E1 = -self.J * (torch.sum(1 * neighbors))

        return 1.0 / (1.0 + torch.exp(E1 - E0))

    def Gibbs_sampling(self, iter=3, init=None):

        sample = self.random_x() if init == None else init

        for epoch in range(iter):
            for i in range(self.n):
                for j in range(self.n):
                    sample[i, j] = -1
                    p0 = self.unnormalized_p(sample)
                    sample[i, j] = 1
                    p1 = self.unnormalized_p(sample)
                    sample[i, j] = torch.bernoulli(p1 / (p0 + p1)) * 2 - 1
            print(self.unnormalized_p(sample).item())

        return sample

    def stable_Gibbs_sampling(self, iter=3, init=None):
        sample = self.random_x() if init == None else init
        samples = [sample.clone()]
        for epoch in range(iter):
            for i in range(self.n):
                for j in range(self.n):
                    p = self._stable_prob(sample, (i, j))
                    sample[i, j] = torch.bernoulli(p) * 2 - 1
            # print(self.unnormalized_p(sample).item())
            samples.append(sample.clone())

        return samples

    def stable_effient_Gibbs_sampling(self, iter=3, init=None):
        sample = self.random_x() if init == None else init
        samples = [sample.clone()]
        for epoch in range(iter):
            for i in range(self.n):
                for j in range(self.n):
                    p = self._stable_effient_prob(sample, (i, j))
                    sample[i, j] = torch.bernoulli(p) * 2 - 1
            # print(self.unnormalized_p(sample).item())
            samples.append(sample.clone())

        return samples


import time


class Timer:
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.start = time.perf_counter()
        return self  # 可用于记录多个时间点

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.perf_counter()
        self.interval = self.end - self.start
        name = f"'{self.name}' " if self.name else ""
        print(f"⏱️ Timer {name}elapsed: {self.interval:.6f} seconds")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 若使用多卡
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    ising = Ising(50, 5)

    with Timer("1"):
        set_seed()
        init = ising.random_x()
        a = ising.stable_Gibbs_sampling(5, init)

    with Timer("2"):
        set_seed()
        init = ising.random_x()
        b = ising.stable_effient_Gibbs_sampling(5, init)

    for i in range(len(a)):
        print(torch.equal(a[i], b[i]))
