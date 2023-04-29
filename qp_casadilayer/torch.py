import casadi as cs
import numpy as np
import torch


class QP_CasadiLayer(torch.nn.Module):
    """A differentiable quadratic programing (QP) layer

    A QP_CasadiLayer solves a parametrized QP problem given by.
    It solves the problem in its forward pass, and it computes
    the derivative of problem's solution map with respect to the parameters in
    its backward pass.

    # Canonical form of quadratic program is
    # min_x 1/2 x'Qx+ q'x, s.t. Ax+b =0 and Gz+h<=0

    Examples:
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


    """

    def __init__(self, n_dim, n_equ=0, n_inequ=0,
                 Q=None, q=None, A=None, b=None, G=None, h=None,
                 solver=None):

        """
        Canonical form of quadratic program is

        min_x 1/2 x'Qx+ q'x, s.t. Ax+b =0 and Gz+h<=0

        Args:
            n_dim: dimension of QP variable in QP canonical form
            n_equ: number of equality constraints in QP canonical form
            n_inequ: number of inequality constraints in QP canonical form
            Q: None if you want it learnable, otherwise set a constant Numpy matrix
            q: None if you want it learnable, otherwise set a constant Numpy matrix
            A: None if you want it learnable, otherwise set a constant Numpy matrix
            b: None if you want it learnable, otherwise set a constant Numpy matrix
            G: None if you want it learnable, otherwise set a constant Numpy matrix
            h: None if you want it learnable, otherwise set a constant Numpy matrix
            solver: QP solver used (TODO)
        """

        super(QP_CasadiLayer, self).__init__()

        # Dimensions in a single QP
        self.n_dim = n_dim
        self.n_equ = n_equ
        self.n_inequ = n_inequ

        # Specified QP solvers. TODO
        self.solver_name = solver

        # Define the casadi symbolic variable in QP
        x = cs.SX.sym('QP_x', self.n_dim)

        # Define the matrices in QP.
        # If a matrix is None in arguments, it means this matrix is learnable
        # NOTE: CasAdi's matrix construction is column-first.
        # To make it compatible to Numpy or Tensor, we need to do transpose when necessary.
        param = []
        if Q is not None:
            Q = cs.DM(Q)
        else:
            Q = cs.SX.sym('QP_Q', self.n_dim, self.n_dim)
            param.append(cs.vec(Q.T))

        if q is not None:
            q = cs.DM(q)
        else:
            q = cs.SX.sym('QP_q', self.n_dim)
            param.append(q)

        if A is not None:
            A = cs.DM(A)
        else:
            A = cs.SX.sym('QP_A', self.n_equ, self.n_dim)
            param.append(cs.vec(A.T))

        if b is not None:
            b = cs.DM(b)
        else:
            b = cs.SX.sym('QP_b', self.n_equ)
            param.append(b)

        if G is not None:
            G = cs.DM(G)
        else:
            G = cs.SX.sym('QP_G', self.n_inequ, self.n_dim)
            param.append(cs.vec(G.T))

        if h is not None:
            h = cs.DM(h)
        else:
            h = cs.SX.sym('QP_h', self.n_inequ)
            param.append(h)

        # vectorize the learnable parameters
        param = cs.vcat(param)
        self.n_param = param.numel()

        # define the symbolic objective in QP
        obj = 1 / 2 * cs.dot(x, Q @ x) + cs.dot(q, x)

        # define the symbolic equality and inequality in QP
        equ = A @ x + b
        inequ = G @ x + h
        constraint = cs.vertcat(equ, inequ)

        # establish QP solver
        quadprog = {'x': x, 'f': obj, 'g': constraint, 'p': param}
        # opts = {'error_on_fail': False}
        opts = {'printLevel': 'none', 'error_on_fail': False}
        self.qp_solver_fn = cs.qpsol('qp_solver_fn', 'qpoases', quadprog, opts)
        self.qp_constraint_lb = cs.vcat(self.n_equ * [0.0] + self.n_inequ * [-cs.inf])
        self.qp_constraint_ub = cs.vcat(self.n_equ * [0.0] + self.n_inequ * [0.0])
        # print(self.qp_solver_fn)

        # establish the backprop through QP using implicit function theorem
        # lagrangian multiplier
        lam_equ = cs.SX.sym('QP_lam_equ', self.n_equ)
        lam_inequ = cs.SX.sym('QP_lam_inequ', self.n_inequ)
        lam_constraint = cs.vertcat(lam_equ, lam_inequ)
        # grad of lagrangian
        lag = obj + cs.dot(lam_equ, equ) + cs.dot(lam_inequ, inequ)
        grad_lag = cs.gradient(lag, x)
        # complementarity
        comple = cs.diag(lam_inequ) @ inequ
        # implicit function
        imp_g = cs.vertcat(grad_lag, equ, comple)
        imp_x = cs.vertcat(x, lam_equ, lam_inequ)
        # implicit theorem
        grad_imp_x = cs.jacobian(imp_g, imp_x)
        grad_imp_p = cs.jacobian(imp_g, param)
        grad_imp_x_p = -cs.inv(grad_imp_x) @ grad_imp_p
        grad_x_p = grad_imp_x_p[0:self.n_dim, :]

        # establish implicit function theorem solver
        self.qp_imp_fn = cs.Function('qp_imp_fn', [x, lam_constraint, param], [grad_x_p])

        # defining new autograd function
        class _diff_qp_layer(torch.autograd.Function):

            @staticmethod
            def forward(ctx, *params):
                # Always suppose the first dimension is always a batch dimension
                # params should be the batch of vectorized matrices (vectors) of QP
                qp_param, = params

                # infer dtype and device
                ctx.dtype = qp_param.dtype
                ctx.device = qp_param.device

                # Solve QP in batch
                qp_param = self.to_numpy(qp_param)
                sol = self.qp_solver_fn(p=qp_param.T, lbg=self.qp_constraint_lb, ubg=self.qp_constraint_ub)
                sol_x = sol['x']
                lam_g = sol['lam_g']
                ctx.qp_param = qp_param
                ctx.sol_x = sol_x
                ctx.lam_g = lam_g

                return self.to_torch(sol_x.full().T, ctx.dtype, ctx.device).squeeze()

            @staticmethod
            def backward(ctx, *grad_outputs):
                # st = time.time()
                backprop_grad = grad_outputs[0]  # shape: [batch x n_dim]
                # Apply the implicit function theorem
                jac_x_p = self.qp_imp_fn(ctx.sol_x, ctx.lam_g,
                                         ctx.qp_param.T).full()  # shape: [n_dim x (n_param*batch_size)]

                jac_x_p = np.array(np.hsplit(jac_x_p, ctx.sol_x.shape[1]))  # shape: [batch x n_dim x n_param]
                jac_x_p = self.to_torch(jac_x_p, ctx.dtype, ctx.device)

                # Assemble the backpropagation
                # grad_p = torch.matmul(backprop_grad.unsqueeze(dim=-2), jac_x_p).squeeze()
                grad_p = torch.einsum('...i,...ij->...j', backprop_grad, jac_x_p)

                return grad_p

        self.diff_qp_layer = _diff_qp_layer.apply

    def forward(self, param):
        """

        Args:
            param: flattened-then-concatenated parameter for QP.
            The order in param is like: param=concatenate(Q.flatten(), q, A.flatten(), b, G.flatten(), h).
            If any matrix are not learnable, as specified by user, it will be missing in param

        Returns:
            solution of QP parameterized by param.

        """

        sol = self.diff_qp_layer(param)

        return sol

    @staticmethod
    def to_numpy(x):
        # convert torch tensor to numpy array
        return x.cpu().detach().double().numpy()

    @staticmethod
    def to_torch(x, dtype, device):
        # convert numpy array to torch tensor
        return torch.from_numpy(x).type(dtype).to(device)
