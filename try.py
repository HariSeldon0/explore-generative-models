import torch

P = [
    [1.0, 0.0, 0.0, 0.0],
    [0.2, 0.3, 0.4, 0.1],
    [0.0, 0.2, 0.0, 0.8],
    [0.0, 0.0, 0.0, 1.0],
]

M = torch.tensor(P)
X0 = torch.tensor([0.0, 1.0, 0.0, 0.0])
T = 30

print()
for t in range(T):
    print(torch.matmul(X0, torch.matrix_power(M, t + 1)))
    # print(torch.matrix_power(M, t + 1))
