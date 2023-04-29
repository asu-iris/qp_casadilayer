import cvxpy as cp
import numpy as np
import torch
from cvxpylayers.torch import CvxpyLayer
import time

torch.manual_seed(0)

n = 5
m = 3
batch_size = 4

P_sqrt = cp.Parameter((n, n), name='P_sqrt')
q = cp.Parameter((n, 1), name='q')
A = cp.Parameter((m, n), name='A')
b = cp.Parameter((m, 1), name='b')

x = cp.Variable((n, 1), name='x')

objective = 0.5 * cp.sum_squares(P_sqrt @ x) + q.T @ x
constraints = [A @ x + b == 0]
prob = cp.Problem(cp.Minimize(objective), constraints)

prob_tch = CvxpyLayer(prob, [P_sqrt, q, A, b], [x])

P_sqrt_tch = torch.randn(batch_size, n, n, requires_grad=True)
q_tch = torch.randn(batch_size, n, 1, requires_grad=True)
A_tch = torch.randn(batch_size, m, n, requires_grad=True)
b_tch = torch.randn(batch_size, m, 1, requires_grad=True)

# forward
st = time.time()
sol_tch, = prob_tch(P_sqrt_tch, q_tch, A_tch, b_tch)
print('forward time', time.time() - st)
print(sol_tch.squeeze())

# backward
st = time.time()
sol_tch.sum().backward()
print('backward time:', time.time() - st)

# grad
print(P_sqrt_tch.grad)
# print(q_tch.grad)
# print(A_tch.grad)
# print(b_tch.grad)

# -------------------------------------------------
P_sqrt_tch.grad.zero_()
q_tch.grad.zero_()
A_tch.grad.zero_()
b_tch.grad.zero_()

from qp_casadilayer.torch import QP_CasadiLayer

qp_layer = QP_CasadiLayer(n_dim=n, n_equ=m)

# compute each matrix
P_tch = torch.matmul(torch.transpose(P_sqrt_tch, -2, -1), P_sqrt_tch)
# concatenate
param_tch = torch.hstack(
    (torch.flatten(P_tch, start_dim=-2),
     torch.flatten(q_tch, start_dim=-2),
     torch.flatten(A_tch, start_dim=-2),
     torch.flatten(b_tch, start_dim=-2))
)

# forward
st = time.time()
qp_sol_tch = qp_layer(param_tch)
print('our forward time:', time.time() - st)
print(qp_sol_tch)

# backward
st = time.time()
qp_sol_tch.sum().backward()
print('our backward time:', time.time() - st)

print(P_sqrt_tch.grad)
# print(q_tch.grad)
# print(A_tch.grad)
# print(b_tch.grad)
