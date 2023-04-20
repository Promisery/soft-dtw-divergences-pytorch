"""Pytorch implementation of the following paper.

Differentiable Divergences between Time Series
Mathieu Blondel, Arthur Mensch, Jean-Philippe Vert
https://arxiv.org/abs/2010.08354
"""

import functools
import torch


@torch.jit.script
def _soft_min_argmin(a, b, c):
    """Computes the soft min and argmin of (a, b, c).

    Args:
    a: scalar value.
    b: scalar value.
    c: scalar value.
    Returns:
    softmin, softargmin[0], softargmin[1], softargmin[2]
    """
    min_abc = torch.min(a, torch.min(b, c))
    exp_a = torch.exp(min_abc - a)
    exp_b = torch.exp(min_abc - b)
    exp_c = torch.exp(min_abc - c)
    s = exp_a + exp_b + exp_c
    exp_a /= s
    exp_b /= s
    exp_c /= s
    val = min_abc - torch.log(s)
    return val, exp_a, exp_b, exp_c


@torch.jit.script
def _sdtw_C(C, V, P):
    """SDTW dynamic programming recursion.

    Args:
    C: cost matrix (input).
    V: intermediate values (output).
    P: transition probability matrix (output).
    """
    B, size_X, size_Y = C.shape

    for i in range(1, size_X + 1):
        for j in range(1, size_Y + 1):

            smin, P[:, i, j, 0], P[:, i, j, 1], P[:, i, j, 2] = \
                _soft_min_argmin(V[:, i, j - 1], V[:, i - 1, j - 1], V[:, i - 1, j])

            # The cost matrix C is indexed starting from 0.
            V[:, i, j] = C[:, i - 1, j - 1] + smin

def sdtw_C(C, gamma=1.0, return_all=False):
    """Computes the soft-DTW value from a cost matrix C.

    Args:
    C: cost matrix, pytorch tensor of shape (size_X, size_Y).
    gamma: regularization strength (scalar value).
    return_all: whether to return intermediate computations.
    Returns:
    sdtw_value if not return_all
    V (intermediate values), P (transition probability matrix) if return_all
    """
    B, size_X, size_Y = C.shape

    # Handle regularization parameter 'gamma'.
    if gamma != 1.0:
        C = C / gamma

    # Matrix containing the values of sdtw.
    V = torch.zeros((B, size_X + 1, size_Y + 1), device=C.device, dtype=C.dtype)
    V[:, :, 0] = 1e10
    V[:, 0, :] = 1e10
    V[:, 0, 0] = 0

    # Tensor containing the probabilities of transition.
    P = torch.zeros((B, size_X + 2, size_Y + 2, 3), device=C.device, dtype=C.dtype)

    _sdtw_C(C, V, P)

    if return_all:
        return V * gamma, P
    else:
        return V[:, size_X, size_Y] * gamma


def sdtw(X, Y, gamma=1.0, return_all=False):
  """Computes the soft-DTW value from time series X and Y.

  The cost is assumed to be the squared Euclidean one.

  Args:
    X: time series, pytorch tensor of shape (size_X, num_dim).
    Y: time series, pytorch tensor of shape (size_Y, num_dim).
    gamma: regularization strength (scalar value).
    return_all: whether to return intermediate computations.
  Returns:
    sdtw_value if not return_all
    V (intermediate values), P (transition probability matrix) if return_all
  """
  C = squared_euclidean_cost(X, Y)
  return sdtw_C(C, gamma=gamma, return_all=return_all)


@torch.jit.script
def _sdtw_grad_C(P, E):
    """Backward dynamic programming recursion.

    Args:
    P: transition probability matrix (input).
    E: expected alignment matrix (output).
    """

    for j in range(E.shape[1] - 2, 0, -1):
        for i in range(E.shape[1] - 2, 0, -1):

            E[:, i, j] = P[:, i, j + 1, 0] * E[:, i, j + 1] + \
                    P[:, i + 1, j + 1, 1] * E[:, i + 1, j + 1] + \
                    P[:, i + 1, j, 2] * E[:, i + 1, j]


def sdtw_grad_C(P, return_all=False):
    """Computes the soft-DTW gradient w.r.t. the cost matrix C.

    The gradient is equal to the expected alignment under the Gibbs distribution.

    Args:
    P: transition probability matrix.
    return_all: whether to return intermediate computations.
    Returns:
    E (expected alignment) if not return_all
    E with edges if return_all
    """
    E = torch.zeros((P.shape[0], P.shape[1], P.shape[2]), device=P.device, dtype=P.dtype)
    E[:, -1, :] = 0
    E[:, :, -1] = 0
    E[:, -1, -1] = 1
    P[:, -1, -1] = 1

    _sdtw_grad_C(P, E)

    if return_all:
        return E
    else:
        return E[:, 1:-1, 1:-1]


def sdtw_value_and_grad_C(C, gamma=1.0):
  """Computes the soft-DTW value *and* gradient w.r.t. the cost matrix C.

  Args:
    C: cost matrix, pytorch tensor of shape (size_X, size_Y).
    gamma: regularization strength (scalar value).
  Returns:
    sdtw_value, sdtw_gradient_C
  """
  size_X, size_Y = C.shape
  V, P = sdtw_C(C, gamma=gamma, return_all=True)
  return V[size_X, size_Y], sdtw_grad_C(P)


def sdtw_value_and_grad(X, Y, gamma=1.0):
  """Computes soft-DTW value *and* gradient w.r.t. time series X.

  The cost is assumed to be the squared Euclidean one.

  Args:
    X: time series, pytorch tensor of shape (size_X, num_dim).
    Y: time series, pytorch tensor of shape (size_Y, num_dim).
    gamma: regularization strength (scalar value).
  Returns:
    sdtw_value, sdtw_gradient_X
  """
  C = squared_euclidean_cost(X, Y)
  val, grad = sdtw_value_and_grad_C(C, gamma=gamma)
  return val, squared_euclidean_cost_vjp(X, Y, grad)


@torch.jit.script
def _sdtw_directional_derivative_C(P, Z, V_dot):
    """Recursion for computing the directional derivative in the direction of Z.

    Args:
    P: transition probability matrix (input).
    Z: direction matrix (input).
    V_dot: intermediate computations (output).
    """
    B, size_X, size_Y = Z.shape
    
    for i in range(1, size_X + 1):
        for j in range(1, size_Y + 1):
            # The matrix Z is indexed starting from 0.
            V_dot[:, i, j] = Z[:, i - 1, j - 1] + \
                        P[:, i, j, 0] * V_dot[:, i, j - 1] + \
                        P[:, i, j, 1] * V_dot[:, i - 1, j - 1] + \
                        P[:, i, j, 2] * V_dot[:, i - 1, j]


def sdtw_directional_derivative_C(P, Z, return_all=False):
    """Computes the soft-DTW directional derivative in the direction of Z.

    Args:
    P: transition probability matrix.
    Z: direction matrix.
    return_all: whether to return intermediate computations.
    Returns:
    sdtw_directional_derivative if not return_all
    V_dot (intermediate values) if return_all
    """
    B, size_X, size_Y = Z.shape

    if size_X != P.shape[1] - 2 or size_Y != P.shape[2] - 2:
        raise ValueError("Z should have shape " + str((P.shape[1], P.shape[2])))

    V_dot = torch.zeros((B, size_X + 1, size_Y + 1), device=P.device, dtype=P.dtype)
    V_dot[:, 0, 0] = 0

    _sdtw_directional_derivative_C(P, Z, V_dot)

    if return_all:
        return V_dot
    else:
        return V_dot[size_X, size_Y]


@torch.jit.script
def _sdtw_hessian_product_C(P, P_dot, E, E_dot, V_dot):
    """Recursion for computing the Hessian product with Z.

    Args:
    P: transition probability matrix (input).
    P_dot: intermediate computations (output).
    E: output of sdtw_grad_C (input).
    E_dot: intermediate computations (output).
    V_dot: output of sdtw_directional_derivative_C (input).
    """
    # Equivalent to using reversed (not currently supported by Numba).

    for j in range(V_dot.shape[2] - 1, 0, -1):
        for i in range(V_dot.shape[1] - 1, 0, -1):

            inner = P[:, i, j, 0] * V_dot[:, i, j - 1]
            inner += P[:, i, j, 1] * V_dot[:, i - 1, j - 1]
            inner += P[:, i, j, 2] * V_dot[:, i - 1, j]

            P_dot[:, i, j, 0] = P[:, i, j, 0] * inner
            P_dot[:, i, j, 1] = P[:, i, j, 1] * inner
            P_dot[:, i, j, 2] = P[:, i, j, 2] * inner

            P_dot[:, i, j, 0] -= P[:, i, j, 0] * V_dot[:, i, j - 1]
            P_dot[:, i, j, 1] -= P[:, i, j, 1] * V_dot[:, i - 1, j - 1]
            P_dot[:, i, j, 2] -= P[:, i, j, 2] * V_dot[:, i - 1, j]

            E_dot[:, i, j] = P_dot[:, i, j + 1, 0] * E[:, i, j + 1] + \
                        P[:, i, j + 1, 0] * E_dot[:, i, j + 1] + \
                        P_dot[:, i + 1, j + 1, 1] * E[:, i + 1, j + 1] + \
                        P[:, i + 1, j + 1, 1] * E_dot[:, i + 1, j + 1] + \
                        P_dot[:, i + 1, j, 2] * E[:, i + 1, j] + \
                        P[:, i + 1, j, 2] * E_dot[:, i + 1, j]


def sdtw_hessian_product_C(P, E, V_dot):
    """Computes the soft-DTW Hessian product.

    Args:
    P: transition probability matrix.
    E: expected alignment matrix (output of sdtw_grad_C).
    V_dot: output of sdtw_directional_derivative_C.
    Returns:
    sdtw_Hessian_product
    """
    E_dot = torch.zeros_like(E)
    P_dot = torch.zeros((E.shape[0], E.shape[1], E.shape[2], 3), device=P.device, dtype=P.dtype)

    if P.shape[0] != E.shape[0] or P.shape[1] != E.shape[1] or P.shape[2] != E.shape[2]:
        raise ValueError("P and E have incompatible shapes.")

    if P.shape[0] != V_dot.shape[0] or P.shape[1] - 1 != V_dot.shape[1] or P.shape[2] - 1 != V_dot.shape[2]:
        raise ValueError("P and V_dot have incompatible shapes.")

    _sdtw_hessian_product_C(P, P_dot, E, E_dot, V_dot)

    return E_dot[:, 1:-1, 1:-1]


def sdtw_entropy_C(C, gamma=1.0):
  """Computes the entropy of the Gibbs distribution associated with soft-DTW.

  Args:
    C: cost matrix, pytorch tensor of shape (size_X, size_Y).
    gamma: regularization strength (scalar value).
  Returns:
    entropy_value
  """
  val, E = sdtw_value_and_grad_C(C, gamma=gamma)
  return (torch.vdot(E.flatten(), C.flatten()) - val) / gamma


def sdtw_entropy(X, Y, gamma=1.0):
  """Computes the entropy of the Gibbs distribution associated with soft-DTW.

  Args:
    X: time series, pytorch tensor of shape (size_X, num_dim).
    Y: time series, pytorch tensor of shape (size_Y, num_dim).
    gamma: regularization strength (scalar value).
  Returns:
    entropy_value
  """
  C = squared_euclidean_cost(X, Y)
  return sdtw_entropy_C(C, gamma=gamma)


def sharp_sdtw_C(C, gamma=1.0):
  """Computes the sharp soft-DTW value from a cost matrix C.

  Args:
    C: cost matrix, pytorch tensor of shape (size_X, size_Y).
    gamma: regularization strength (scalar value).
  Returns:
    sharp_sdtw_value
  """
  P = sdtw_C(C, gamma=gamma, return_all=True)[1]
  return sdtw_directional_derivative_C(P, C)


def sharp_sdtw(X, Y, gamma=1.0):
  """Computes the sharp soft-DTW value from time series X and Y.

  The cost is assumed to be the squared Euclidean one.

  Args:
    X: time series, pytorch tensor of shape (size_X, num_dim).
    Y: time series, pytorch tensor of shape (size_Y, num_dim).
    gamma: regularization strength (scalar value).
  Returns:
    sharp_sdtw_value
  """
  C = squared_euclidean_cost(X, Y)
  return sharp_sdtw_C(C, gamma=gamma)


def sharp_sdtw_value_and_grad_C(C, gamma=1.0):
    """Computes the sharp soft-DTW value *and* its gradient w.r.t. C.

    Args:
        C: cost matrix, pytorch tensor of shape (size_X, size_Y).
        gamma: regularization strength (scalar value).
    Returns:
        sharp_sdtw_value, sharp_sdtw_grad_C
    """
    V, P = sdtw_C(C, gamma=gamma, return_all=True)
    E = sdtw_grad_C(P, return_all=True)
    V_dot = sdtw_directional_derivative_C(P, C, return_all=True)
    HC = sdtw_hessian_product_C(P, E, V_dot)
    G = E[:, 1:-1, 1:-1]
    val = V_dot[:, -1, -1]
    grad = G + HC / gamma
    return val, grad


def sharp_sdtw_value_and_grad(X, Y, gamma=1.0):
  """Computes the sharp soft-DTW value *and* its gradient w.r.t. X.

  The cost is assumed to be the squared Euclidean one.

  Args:
    X: time series, pytorch tensor of shape (size_X, num_dim).
    Y: time series, pytorch tensor of shape (size_Y, num_dim).
    gamma: regularization strength (scalar value).
  Returns:
    sharp_sdtw_value, sharp_sdtw_grad_X
  """
  C = squared_euclidean_cost(X, Y)
  val, grad = sharp_sdtw_value_and_grad_C(C, gamma=gamma)
  return val, squared_euclidean_cost_vjp(X, Y, grad)


def squared_euclidean_cost(X, Y, return_all=False, log=False):
    """Computes the squared Euclidean cost.

    Args:
    X: time series, pytorch tensor of shape (size_X, num_dim).
    Y: time series, pytorch tensor of shape (size_Y, num_dim).
    return_all: whether to also return the cost matrices for (X, X) and (Y, Y).
    log: whether to use the log-augmented cost or not (see paper).
    Returns:
    C(X, Y) if not return_all
    C(X, Y), C(X, X), C(Y, Y) if return_all
    """
    def _C(C):
        if log:
            return C + torch.log(2 - torch.exp(-C))
        else:
            return C

    X_sqnorms = torch.sum(X ** 2, dim=-1) * 0.5
    Y_sqnorms = torch.sum(Y ** 2, dim=-1) * 0.5
    XY = torch.matmul(X, Y.transpose(-1, -2)).to(X_sqnorms)

    if return_all:
        C_XY = -XY
        C_XY += X_sqnorms.unsqueeze(-1)
        C_XY += Y_sqnorms.unsqueeze(-2)

        C_XX = -torch.matmul(X, X.transpose(-1, -2))
        C_XX += X_sqnorms.unsqueeze(-1)
        C_XX += X_sqnorms.unsqueeze(-2)

        C_YY = -torch.matmul(Y, Y.transpose(-1, -2))
        C_YY += Y_sqnorms.unsqueeze(-1)
        C_YY += Y_sqnorms.unsqueeze(-2)

        return _C(C_XY), _C(C_XX), _C(C_YY)

    else:
        C = -XY
        C += X_sqnorms.unsqueeze(-1)
        C += Y_sqnorms.unsqueeze(-2)
        return _C(C)


def squared_euclidean_cost_vjp(X, Y, E, log=False):
    """Left-product with the Jacobian of the squared Euclidean cost.

    Args:
    X: time series, pytorch tensor of shape (size_X, num_dim).
    Y: time series, pytorch tensor of shape (size_Y, num_dim).
    E: matrix to multiply with, pytorch tensor of shape (size_X, size_Y).
    log: whether to use the log-augmented cost or not (see paper).
    Returns:
    vjp
    """
    if E.shape[1] != X.shape[1] or E.shape[2] != Y.shape[1]:
        raise ValueError("E.shape should be equal to (len(X), len(Y)).")

    e = E.sum(axis=2)
    vjp = X * e.unsqueeze(-1)
    vjp -= torch.matmul(E, Y)

    if log:
        C = squared_euclidean_cost(X, Y)
        deriv = torch.exp(-C) / (2 - torch.exp(-C))
        vjp += squared_euclidean_cost_vjp(X, Y, E * deriv)

    return vjp


def squared_euclidean_cost_jvp(X, Y, Z):
  """Right-product with the Jacobian of the squared Euclidean cost.

  Args:
    X: time series, pytorch tensor of shape (size_X, num_dim).
    Y: time series, pytorch tensor of shape (size_Y, num_dim).
    Z: matrix to multiply with, pytorch tensor of shape (size_X, num_dim).
  Returns:
    jvp
  """
  if Z.shape[0] != X.shape[0] or Z.shape[1] != X.shape[1]:
    raise ValueError("Z should be of the same shape as X.")

  if Y.shape[1] != Z.shape[1]:
    raise ValueError("Y.shape[1] should be equal to Z.shape[1].")

  jvp = -torch.matmul(Z, Y.T)
  jvp += torch.sum(X * Z, dim=1)[:, None]
  return jvp


def squared_euclidean_distance(X, Y):
  """Computes the squared Euclidean distance between two time series.

  The two time series must have the same length.

  Args:
    X: time series, pytorch tensor of shape (size_X, num_dim).
    Y: time series, pytorch tensor of shape (size_Y, num_dim).
  Returns:
    distance_value
  """
  if len(X) != len(Y) or X.shape[1] != Y.shape[1]:
    raise ValueError("X and Y have incompatible shapes.")

  return torch.sum((X - Y) ** 2) * 0.5


def _divergence(func, X, Y):
  """Converts a value function into a divergence.

  The cost is assumed to be the squared Euclidean one.

  Args:
    func: function to use.
    X: time series, pytorch tensor of shape (size_X, num_dim).
    Y: time series, pytorch tensor of shape (size_Y, num_dim).
  Returns:
    func(C(X,Y)) - 0.5 * func(C(X,X)) - 0.5 * func(C(Y,Y))
  """
  C_XY, C_XX, C_YY = squared_euclidean_cost(X, Y, return_all=True)
  value = func(C_XY)
  value -= func(C_XX) * 0.5
  value -= func(C_YY) * 0.5
  return value


def _divergence_value_and_grad(func, X, Y):
    """Converts a value and grad function into a divergence.

    The cost is assumed to be the squared Euclidean one.

    Args:
    func: function to use.
    X: time series, pytorch tensor of shape (size_X, num_dim).
    Y: time series, pytorch tensor of shape (size_Y, num_dim).
    Returns:
    div_value, div_grad_X
    """
    C_XY, C_XX, C_YY = squared_euclidean_cost(X, Y, return_all=True)
    value_XY, grad_XY = func(C_XY)
    value_XX, grad_XX = func(C_XX)
    value_YY, grad_YY = func(C_YY)
    value = value_XY - value_XX * 0.5 - value_YY * 0.5
    grad = squared_euclidean_cost_vjp(X, Y, grad_XY)
    # The 0.5 factor cancels out.
    grad -= squared_euclidean_cost_vjp(X, X, grad_XX)
    return value, grad


def sdtw_div(X, Y, gamma=1.0):
  """Compute the soft-DTW divergence value between time series X and Y.

  The cost is assumed to be the squared Euclidean one.

  Args:
    X: time series, pytorch tensor of shape (size_X, num_dim).
    Y: time series, pytorch tensor of shape (size_Y, num_dim).
    gamma: regularization strength (scalar value).
  Returns:
    divergence_value
  """
  func = functools.partial(sdtw_C, gamma=gamma)
  return _divergence(func, X, Y)


def sdtw_div_value_and_grad(X, Y, gamma=1.0):
  """Compute the soft-DTW divergence value *and* gradient w.r.t. X.

  The cost is assumed to be the squared Euclidean one.

  Args:
    X: time series, pytorch tensor of shape (size_X, num_dim).
    Y: time series, pytorch tensor of shape (size_Y, num_dim).
    gamma: regularization strength (scalar value).
  Returns:
    divergence_value, divergence_grad
  """
  func = functools.partial(sdtw_value_and_grad_C, gamma=gamma)
  return _divergence_value_and_grad(func, X, Y)


def sharp_sdtw_div(X, Y, gamma=1.0):
  """Compute the sharp soft-DTW divergence value between time series X and Y.

  The cost is assumed to be the squared Euclidean one.

  Args:
    X: time series, pytorch tensor of shape (size_X, num_dim).
    Y: time series, pytorch tensor of shape (size_Y, num_dim).
    gamma: regularization strength (scalar value).
  Returns:
    divergence_value
  """
  func = functools.partial(sharp_sdtw_C, gamma=gamma)
  return _divergence(func, X, Y)


def sharp_sdtw_div_value_and_grad(X, Y, gamma=1.0):
    """Compute the sharp soft-DTW divergence value *and* gradient w.r.t. X.

    The cost is assumed to be the squared Euclidean one.

    Args:
    X: time series, pytorch tensor of shape (size_X, num_dim).
    Y: time series, pytorch tensor of shape (size_Y, num_dim).
    gamma: regularization strength (scalar value).
    Returns:
    divergence_value, divergence_grad
    """
    func = functools.partial(sharp_sdtw_value_and_grad_C, gamma=gamma)
    return _divergence_value_and_grad(func, X, Y)
