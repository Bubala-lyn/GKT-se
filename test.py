import torch

x = torch.randn((32, 59))

y = torch.randn((32, 59))

z = torch.ones_like(x)

for x_index in range(x.size()[0]):
    for y_index in range(y.size()[1]):
        z[x_index][y_index] = x[x_index][y_index] * y[x_index][y_index]

result = (x * y).sum(dim=1)

print(torch.equal(z, result))
