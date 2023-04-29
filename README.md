# Quadratic Programming Layer (Pytorch)

This implements a differentiable quadratic programing layer (QP_CasadiLayer),
which can be embedded into any neural networks written in Pytorch.
A QP_CasadiLayer solves a parametrized QP of the following
canonical form:

min_x 1/2 x'Qx+ q'x, s.t. Ax+b =0 and Gz+h<=0

In forward pass, this QP_CasadiLayer solves for the solution to a QP  using [CasAdi](https://web.casadi.org/docs/) 
qpsol API. In  backward pass, the QP_CasadiLayer computes the derivative of the QP solution 
with respect to the QP parameters using Implicit Function Theorem.

### Some unofficial features

- Supports batching process of QPs
- Initial testing shows it is >50x faster than [cvxpylayers](https://github.com/cvxgrp/cvxpylayers) in forward pass
- Initial testing shows it is >10x faster than [cvxpylayers](https://github.com/cvxgrp/cvxpylayers) in backward pass


### An Example

```
import torch
from qp_casadilayer.torch import QP_CasadiLayer

n_dim = 6
n_equ = 3
n_inequ = 2
batch_size = 4

P_sqrt_tch = torch.randn(batch_size, n_dim, n_dim, requires_grad=True)
P_tch = torch.matmul(torch.transpose(P_sqrt_tch, -2, -1), P_sqrt_tch)
q_tch = torch.randn(batch_size, n_dim, 1, requires_grad=True)
A_tch = torch.randn(batch_size, n_equ, n_dim, requires_grad=True)
b_tch = torch.randn(batch_size, n_equ, 1, requires_grad=True)
G_tch = torch.randn(batch_size, n_inequ, n_dim, requires_grad=True)
h_tch = torch.randn(batch_size, n_inequ, 1, requires_grad=True)

# vectorize and concatenate all matrices in QP as batch of vectors
param_tch = torch.hstack(
    (torch.flatten(P_tch, start_dim=-2),
     torch.flatten(q_tch, start_dim=-2),
     torch.flatten(A_tch, start_dim=-2),
     torch.flatten(b_tch, start_dim=-2),
     torch.flatten(G_tch, start_dim=-2),
     torch.flatten(h_tch, start_dim=-2),
     )
)

# create a qp layer
qp_layer = QP_CasadiLayer(n_dim=n_dim, n_equ=n_equ, n_inequ=n_inequ)

# forward
qp_sol_tch = qp_layer(param_tch)

# backward
qp_sol_tch.sum().backward()


```

### NOTE
This repo will or will not be maintained since its release. No warrants are offered. 
 Users who seek more comprehensive APIs for differentiable convex layers should be directed to 
the following repo

- cvxpylayers: [https://github.com/cvxgrp/cvxpylayers](https://github.com/cvxgrp/cvxpylayers)
- relevant papers: 
  - [Differentiable Convex Optimization Layers](https://arxiv.org/abs/1910.12430) by Akshay Agrawal, Brandon Amos, Shane Barratt, Stephen Boyd, Steven Diamond, Zico Kolter
  - [Differentiating through log-log convex programs](https://web.stanford.edu/~boyd/papers/pdf/diff_llcvx.pdf) by Agrawal, Akshay and Boyd, Stephen
  - [OptNet: Differentiable Optimization as a Layer in Neural Networks](https://arxiv.org/abs/1703.00443) by Brandon Amos, J. Zico Kolter
