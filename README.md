# Quadratic Programming CasAdi Layer

A differentiable quadratic programing (QP) layer, that can be embedded into any neural networks. A QP_CasadiLayer solves a parametrized QP problem given by
the canonical form of quadratic program is

 min_x 1/2 x'Qx+ q'x, s.t. Ax+b =0 and Gz+h<=0

It solves the problem in its forward pass (using CasAdi QP solving API), and it computes
the derivative of problem's solution map with respect to the parameters in
its backward pass.

### Some features:
- It supports batching
- Testing results show it is >50x faster than [cvxpylayers](https://github.com/cvxgrp/cvxpylayers) in forward pass
- Testing results show it is >10x faster than [cvxpylayers](https://github.com/cvxgrp/cvxpylayers) in backward pass



### Example:
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
