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

    def __init__(self, n_dim, n_equ, n_inequ,
                 diff_Q=False, diff_q=False, diff_A=False, diff_b=False, diff_G=False, diff_h=False,
                 solver=None):

        """
        Canonical form of quadratic program is

        min_x 1/2 x'Qx+ q'x, s.t. Ax+b =0 and Gz+h<=0

        Args:
            n_dim: dimension of QP variable in QP canonical form
            n_equ: number of equality constraints in QP canonical form
            n_inequ: number of inequality constraints in QP canonical form
            diff_Q: True if you want it differentiable, otherwise set as query non-diff tensor
            diff_q: True if you want it differentiable, otherwise set as query non-diff tensor
            diff_A: True if you want it differentiable, otherwise set as query non-diff tensor
            diff_b: True if you want it differentiable, otherwise set as query non-diff tensor
            diff_G: True if you want it differentiable, otherwise set as query non-diff tensor
            diff_h: True if you want it differentiable, otherwise set as query non-diff tensor
            solver: QP solver (TODO)
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
        diff_param = []
        query_param = []

        Q = cs.SX.sym('QP_Q', self.n_dim, self.n_dim)
        if diff_Q:
            diff_param.append(cs.vec(Q.T))
        else:
            query_param.append(cs.vec(Q.T))

        q = cs.SX.sym('QP_q', self.n_dim)
        if diff_q:
            diff_param.append(q)
        else:
            query_param.append(q)

        A = cs.SX.sym('QP_A', self.n_equ, self.n_dim)
        if diff_A:
            diff_param.append(cs.vec(A.T))
        else:
            query_param.append(cs.vec(A.T))

        b = cs.SX.sym('QP_b', self.n_equ)
        if diff_b:
            diff_param.append(b)
        else:
            query_param.append(b)

        G = cs.SX.sym('QP_G', self.n_inequ, self.n_dim)
        if diff_G:
            diff_param.append(cs.vec(G.T))
        else:
            query_param.append(cs.vec(G.T))

        h = cs.SX.sym('QP_h', self.n_inequ)
        if diff_h:
            diff_param.append(h)
        else:
            query_param.append(h)

        # vectorize the learnable parameters
        diff_param = cs.vcat(diff_param)
        self.n_learn_param = diff_param.numel()

        # vectorize the query parameters
        query_param = cs.vcat(query_param)
        self.n_query_param = query_param.numel()

        # concatenate two params
        param = cs.vertcat(diff_param, query_param)
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
        grad_imp_p = cs.jacobian(imp_g, diff_param)
        grad_imp_x_p = -cs.inv(grad_imp_x) @ grad_imp_p
        grad_x_p = grad_imp_x_p[0:self.n_dim, :]

        # establish implicit function theorem solver
        self.qp_imp_fn = cs.Function('qp_imp_fn', [x, lam_constraint, param], [grad_x_p])

        # defining new autograd function
        class _diff_qp_layer(torch.autograd.Function):

            @staticmethod
            def forward(ctx, *params):
                """
                Args:
                    ctx: context for autograd
                    *params: (diff_param, query_param) in canonical QP, where
                    diff_param is [batch_size, n_learn_param (hyperplane_dim*n_hyperplane)]
                    query_param is [batch_size, n_point, n_query_param].

                Returns:
                    Solution of QP, which has a shape [batch_size, n_point, n_dim]
                """

                learn_param, query_param = params

                # infer dtype and device
                ctx.dtype = learn_param.dtype
                ctx.device = learn_param.device

                # determine number of query point
                ctx.n_point = query_param.shape[-2]

                # broadcast diff_param to match the shape of query_param
                learn_param = learn_param.unsqueeze(dim=-2).expand_as(query_param)
                param = torch.cat((learn_param, query_param), dim=-1)  # [batch_size, n_point, n_param]

                # pass param into QP solver and solve it in batch
                sol_param = param.reshape(-1, self.n_param)  # [batch_size*n_point, n_param]
                sol_param = self.to_numpy(sol_param)
                sol = self.qp_solver_fn(p=sol_param.T, lbg=self.qp_constraint_lb, ubg=self.qp_constraint_ub)
                sol_x = sol['x']  # [n_dim, batch_size*n_point]
                sol_lam_g = sol['lam_g']
                ctx.sol_param = sol_param
                ctx.sol_x = sol_x
                ctx.sol_lam_g = sol_lam_g

                return self.to_torch(sol_x.full().T, ctx.dtype, ctx.device).reshape(-1, ctx.n_point, self.n_dim)

            @staticmethod
            def backward(ctx, *grad_outputs):
                backprop_grad = grad_outputs[0]  # [batch_size, n_point, n_dim]

                # Implicit function theorem
                jac_x_p = self.qp_imp_fn(ctx.sol_x, ctx.sol_lam_g,
                                         ctx.sol_param.T).full()  # [n_dim, n_learn_param*batch_size*n_point]

                jac_x_p = np.array(np.hsplit(jac_x_p, ctx.sol_x.shape[1]))  # [batch_size*n_point, n_dim, n_learn_param]

                # reshape to [batch_size, n_point, n_dim, n_learn_param]
                jac_x_p = self.to_torch(jac_x_p, ctx.dtype, ctx.device).reshape(-1, ctx.n_point, self.n_dim,
                                                                                self.n_learn_param)
                # Assemble the backpropagation
                grad_p = torch.einsum('...i,...ij->...j', backprop_grad,
                                      jac_x_p)  # [batch_size, n_point, n_learn_param]

                return grad_p.sum(dim=-2), None

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
