import numpy as np

import torch


def infer(method, X, D, lambd, **kwargs):
    """Run sparse coding inference. This function dispatches different methods.

    Parameters
    ----------
    method : str
        Which type of inference to use.
    X : torch tensor (examples, features)
        Dataset to run inference on.
    D : torch tensor (features, sources)
        Linear generative dictionary.
    lambd : float
        Sparse coefficient.
    args : tuple
        Additional args for `method`.

    Returns
    -------
    A : torch tensor (examples, sources)
        Inferred sparse coefficients.
    """

    if method.lower() == 'fista':
        A = fista(X, D, lambd, **kwargs)
    elif method.lower() == 'lca':
        A = LCA(X, D, lambd, **kwargs)
    else:
        raise ValueError
    return A


def fista(X, D, lambd, max_iter=250, tol=1e-4, verbose=False):
    n_ex, n_feat = X.shape
    n_act = D.shape[0]

    gram = D.t().mm(D)
    L = 2. * torch.symeig(gram)[0][-1]
    zero = torch.tensor(0., dtype=X.dtype, device=X.device)
    yt = torch.zeros((n_ex, n_act), dtype=X.dtype, device=X.device)
    xt = torch.zeros_like(yt)
    t = 1.

    se = torch.sum((X)**2, dim=1).mean().detach().cpu().numpy()
    seo = se

    for ii in range(max_iter):
        diff = X - yt.mm(D)
        sep = torch.sum((diff)**2, dim=1).mean() + lambd * abs(yt).sum(dim=1).mean()
        sep = sep.detach().cpu().numpy()
        if (se - sep) / max(1., max(abs(se), abs(sep))) < tol:
            if ii > 0:
                break
        se = sep
        grad = - (diff).mm(D.t())
        yt -= grad / L
        xtp = torch.max(abs(yt) - lambd / L, zero) * torch.sign(yt)
        t = 0.5 * (1. + np.sqrt(1. + 4 * t**2))
        yt = xt + (t - 1.) * (xtp - xt) / t
        xt = xtp
    if verbose:
        print('inference', ii, seo, sep)
    return xt


def LCA(X, D, lambd, soft=True, max_iter=250, tol=1e-4, verbose=False):
    n_ex, n_feat = X.shape
    n_act = D.shape[0]

    gram = D.mm(D.t())
    L = 2. * torch.symeig(gram)[0][-1]
    eta = 1. / L
    u = torch.zeros((n_ex, n_act), dtype=X.dtype, device=X.device)
    s = torch.zeros_like(u)
    c = gram - torch.eye(n_act, dtype=X.dtype, device=X.device)
    b = X.mm(D.t())
    zero = torch.tensor(0., dtype=X.dtype, device=X.device)

    se = torch.sum((X)**2, dim=1).mean().detach().cpu().numpy()
    seo = se

    for ii in range(max_iter):
        ci = s.mm(c)
        u = eta * (b - ci) + (1. - eta) * u
        if soft:
            s = torch.sign(u) * torch.max(abs(u) - lambd * eta, zero)
            s = torch.sign(u) * torch.max(abs(u) - lambd, zero)
        else:
            s = u * (abs(u) > lambd).type(X.dtype)

        diff = X - s.mm(D)
        sep = torch.sum((diff)**2, dim=1).mean() + lambd * abs(s).sum(dim=1).mean()
        sep = sep.detach().cpu().numpy()
        if (se - sep) / max(1., max(abs(se), abs(sep))) < tol:
            if ii > 0:
                break
        se = sep
    if verbose:
        print('inference', ii, seo, sep)
    return s
