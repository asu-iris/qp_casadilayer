import casadi as cs
import numpy as np
import torch


class QP_CasadiLayer_Query(torch.nn.Module):
    """A differentiable quadratic programing (QP) layer

    A QP_CasadiLayer solves a parametrized QP problem given by.
    It solves the problem in its forward pass, and it computes
    the derivative of problem's solution map with respect to the parameters in
    its backward pass.

    # Canonical form of quadratic program is
    # min_x 1/2 x'Qx+ q'x, s.t. Ax+b =0 and Gz+h<=0

    Examples:

    """

    def __init__(self, n_dim, n_equ=0, n_inequ=0,
                 Q=True, q=True, A=True, b=True, G=True, h=True,
                 solver=None):

        """
        Canonical form of quadratic program is

        min_x 1/2 x'Qx+ q'x, s.t. Ax+b =0 and Gz+h<=0

        Args:
            n_dim: dimension of QP variable in QP canonical form
            n_equ: number of equality constraints in QP canonical form
            n_inequ: number of inequality constraints in QP canonical form
            Q: True if you want it learnable, otherwise set as query tensor (no grad is calculated)
            q: True if you want it learnable, otherwise set as query tensor (no grad is calculated)
            A: True if you want it learnable, otherwise set as query tensor (no grad is calculated)
            b: True if you want it learnable, otherwise set as query tensor (no grad is calculated)
            G: True if you want it learnable, otherwise set as query tensor (no grad is calculated)
            h: True if you want it learnable, otherwise set as query tensor (no grad is calculated)
            solver: QP solver used (TODO)
        """

        super(QP_CasadiLayer_Query, self).__init__()

        # Dimensions in a single QP
        self.n_dim = n_dim
        self.n_equ = n_equ
        self.n_inequ = n_inequ

        # Specified QP solvers. TODO
        self.solver_name = solver

        # Define the casadi symbolic variable in QP
        x = cs.SX.sym('QP_x', self.n_dim)

        # Define the matrices in QP.
        # If a matrix is True in arguments, it means this matrix is learnable
        # NOTE: CasAdi's matrix construction is column-first.
        # To make it compatible to Numpy or Tensor, we need to do transpose when necessary.
        learn_param = []
        query_param = []

        Q = cs.SX.sym('QP_Q', self.n_dim, self.n_dim)
        if Q:
            learn_param.append(cs.vec(Q.T))
        else:
            query_param.append(cs.vec(Q.T))

        q = cs.SX.sym('QP_q', self.n_dim)
        if q:
            learn_param.append(q)
        else:
            query_param.append(q)

        A = cs.SX.sym('QP_A', self.n_equ, self.n_dim)
        if A:
            learn_param.append(cs.vec(A.T))
        else:
            query_param.append(cs.vec(A.T))

        b = cs.SX.sym('QP_b', self.n_equ)
        if b:
            learn_param.append(b)
        else:
            query_param.append(b)

        G = cs.SX.sym('QP_G', self.n_inequ, self.n_dim)
        if G:
            learn_param.append(cs.vec(G.T))
        else:
            query_param.append(cs.vec(G.T))

        h = cs.SX.sym('QP_h', self.n_inequ)
        if h:
            learn_param.append(h)
        else:
            query_param.append(h)

        # vectorize the learnable parameters
        learn_param = cs.vcat(learn_param)
        self.n_learn_param = learn_param.numel()

        # vectorize the query parameters
        query_param = cs.vcat(query_param)
        self.n_query_param = query_param.numel()

        # concatenate two params
        param = cs.vertcat(learn_param, query_param)
        self.n_param = self.n_learn_param + self.n_query_param

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
        grad_imp_p = cs.jacobian(imp_g, learn_param)
        grad_imp_x_p = -cs.inv(grad_imp_x) @ grad_imp_p
        grad_x_p = grad_imp_x_p[0:self.n_dim, :]

        # establish implicit function theorem solver
        self.qp_imp_fn = cs.Function('qp_imp_fn', [x, lam_constraint, param], [grad_x_p])

        # defining new autograd function
        class _diff_qp_layer(torch.autograd.Function):

            @staticmethod
            def forward(ctx, *params):
                # Always suppose the first dimension is always a batch dimension
                """

                Args:
                    ctx: context for autograd
                    *params: (learn_param, query_param) in canonical QP, where
                    learn_param is of [batch_size1 x n_learn_param] and
                    query_param is of [batch_size2 x n_query_param]. When batch_size1 is not equal to batch_size2,
                    we need to do some broadcast, depending on the values of batch_size1 and batch_size2.

                Returns:
                    Solution of QP, which has a shape [batch_size x n_dim], where
                    batch_size will depend on batch_size1 and batch_size2 in the input arguments,
                    according to following rules:
                    if batch_size1 equals batch_size2,  batch_size= batch_size1 (or batch_size2).
                    else, batch_size= (batch_size1 x batch_size2)
                """

                learn_param, query_param = params

                # infer dtype and device
                ctx.dtype = learn_param.dtype
                ctx.device = learn_param.device

                # if any param misses the batch dimension, add it
                if learn_param.ndim == 1:
                    learn_param = torch.unsqueeze(learn_param, 0)
                if query_param.ndim == 1:
                    query_param = torch.unsqueeze(query_param, 0)

                # determine the batch size for both params, and broadcast if necessary.
                ctx.batch_size1 = learn_param.shape[0]
                ctx.batch_size2 = query_param.shape[0]

                # if batch_size1 equals batch_size2, broadcast will not be performed
                if ctx.batch_size1 == ctx.batch_size2:
                    param = torch.hstack((learn_param, query_param))  # [batch_size1 (or batch_size2) x n_param]
                    # pass param into QP solver and solve it in batch!
                    param = self.to_numpy(param)
                    sol = self.qp_solver_fn(p=param.T, lbg=self.qp_constraint_lb, ubg=self.qp_constraint_ub)
                    sol_x = sol['x']  # [n_dim x batch_size1 (or batch_size2)]
                    lam_g = sol['lam_g']
                    ctx.param = param
                    ctx.sol_x = sol_x
                    ctx.lam_g = lam_g
                    return self.to_torch(sol_x.full().T, ctx.dtype, ctx.device).squeeze()

                # if batch_size1 is not equal to batch_size2, broadcast will be performed
                else:
                    learn_param = torch.repeat_interleave(learn_param, ctx.batch_size2, dim=0)
                    query_param = query_param.repeat(ctx.batch_size1, 1)
                    param = torch.hstack((learn_param, query_param))  # [batch_size1 * batch_size2 x n_param]

                    # pass param into QP solver and solve it in batch!
                    param = self.to_numpy(param)
                    sol = self.qp_solver_fn(p=param.T, lbg=self.qp_constraint_lb, ubg=self.qp_constraint_ub)
                    sol_x = sol['x']  # [n_dim x batch_size1 * batch_size2]
                    lam_g = sol['lam_g']
                    ctx.param = param
                    ctx.sol_x = sol_x
                    ctx.lam_g = lam_g

                    return self.to_torch(sol_x.full().T, ctx.dtype, ctx.device).reshape(
                        (ctx.batch_size1, ctx.batch_size2, -1)).squeeze()

            @staticmethod
            def backward(ctx, *grad_outputs):

                if ctx.batch_size1 == ctx.batch_size2:
                    backprop_grad = grad_outputs[0]  # [batch_size1 (or batch_size2), n_dim]
                else:
                    backprop_grad = grad_outputs[0]  # [batch_size1, batch_size2, n_dim]
                    backprop_grad = backprop_grad.reshape((-1, n_dim))  # [batch_size1*batch_size2, n_dim]

                # Apply implicit function theorem (below, batch_size is either batch_size1 or batch_size1*batch_size2)
                jac_x_p = self.qp_imp_fn(ctx.sol_x, ctx.lam_g,
                                         ctx.param.T).full()  # shape: [n_dim x (n_param * batch_size)]

                jac_x_p = np.array(np.hsplit(jac_x_p, ctx.sol_x.shape[1]))  # shape: [batch_size x n_dim x n_param]
                jac_x_p = self.to_torch(jac_x_p, ctx.dtype, ctx.device)

                # Assemble the backpropagation
                # grad_p = torch.matmul(backprop_grad.unsqueeze(dim=-2), jac_x_p).squeeze()
                grad_p = torch.einsum('...i,...ij->...j', backprop_grad, jac_x_p)

                return grad_p, None

        self.diff_qp_layer = _diff_qp_layer.apply

    def forward(self, learn_param, query_param):

        """

        Args:
            learn_param: learnable parameters in canonical QP,  [batch_size1 x n_learn_param]
            query_param: query parameters in canonical QP,  [batch_size1 x n_learn_param]

        Returns:
            Solution of QP, which has a shape [batch_size x n_dim], where
            batch_size will depend on batch_size1 and batch_size2 in the input arguments,
            according to following rules:
            if batch_size1 equals batch_size2,  batch_size= batch_size1 (or batch_size2).
            else, batch_size= (batch_size1 x batch_size2)
        """

        sol = self.diff_qp_layer(learn_param, query_param)

        return sol

    @staticmethod
    def to_numpy(x):
        # convert torch tensor to numpy array
        return x.cpu().detach().double().numpy()

    @staticmethod
    def to_torch(x, dtype, device):
        # convert numpy array to torch tensor
        return torch.from_numpy(x).type(dtype).to(device)
