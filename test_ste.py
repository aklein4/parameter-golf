import torch
import torch.nn.functional as F

from torch import Tensor

import matplotlib.pyplot as plt


def rect_sigmoid(x: Tensor) -> Tensor:
    return F.hardsigmoid(6.0 * x - 3.0)

class STEFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        return (
            rect_sigmoid(x - 0.05)
            - rect_sigmoid(-x - 0.05)
        )


    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tensor:
        return grad_output


def test_ste():
    
    x = torch.linspace(-1.2, 1.2, steps=1000)
    y = STEFunction.apply(x)

    plt.plot(x.detach().numpy(), y.detach().numpy())
    plt.grid()
    plt.savefig("ste.png")


def test_var():

    x = torch.linspace(0.0, 1.0, steps=1000)

    y = torch.sqrt(x * (1.0 - x))

    y_approx = 10 * (0.5 - torch.abs(x - 0.5))

    plt.plot(x.detach().numpy(), y.detach().numpy())
    plt.plot(x.detach().numpy(), y_approx.detach().numpy())
    plt.ylim(0.0, 0.6)
    plt.grid()
    plt.savefig("var.png")


def test_mask():

    d = 4

    q_emb = 1.0 - torch.eye(d)
    k_emb = torch.masked_fill(
        torch.zeros(d, d),
        torch.eye(d, dtype=torch.bool),
        -100.0
    )

    seq = torch.randint(0, 16//d, (16,))
    seq[0] = 1
    print(seq)
    seq_index = ((seq == 1).long().cumsum(-1) - 1).clamp(max=d-1)

    q = q_emb[seq_index]
    k = k_emb[seq_index]

    mat = q @ k.transpose(-2, -1)

    plt.matshow(mat.detach().numpy())
    plt.colorbar()
    plt.savefig("mask.png")


if __name__ == "__main__":
    test_mask()
