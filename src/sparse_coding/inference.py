import warnings
import numpy as np
from sklearn.linear_model import Lasso

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
    elif method.lower() == 'ista':
        A = ista(X, D, lambd, **kwargs)
    elif method.lower() == 'lca':
        A = LCA(X, D, lambd, **kwargs)
    elif method.lower() == 'cd':
        A = cd(X, D, lambd, **kwargs)
    else:
        raise ValueError
    return A


def fista(X, D, lambd, max_iter=250, tol=1e-4, verbose=False,
          return_history=False):
    n_ex, n_feat = X.shape
    n_act = D.shape[0]

    gram = D.t().mm(D)
    L = 2. * torch.linalg.eigvalsh(gram)[-1]
    zero = torch.tensor(0., dtype=X.dtype, device=X.device)
    yt = torch.zeros((n_ex, n_act), dtype=X.dtype, device=X.device)
    xtm = torch.zeros_like(yt)
    if return_history:
        yth = torch.zeros((max_iter, n_ex, n_act), dtype=X.dtype, device=X.device)
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
        xt = yt - grad / L
        xt = torch.max(abs(xt) - lambd / L, zero) * torch.sign(xt)
        t = 0.5 * (1. + np.sqrt(1. + 4 * t**2))
        yt = xt + (t - 1.) * (xt - xtm) / t
        xtm = xt
        if return_history:
            yth[ii] = yt
    if verbose:
        print('inference', ii, seo, sep)
    if return_history:
        return yth[:ii]
    else:
        return yt


def ista(X, D, lambd, max_iter=250, tol=1e-4, verbose=False,
          return_history=False):
    n_ex, n_feat = X.shape
    n_act = D.shape[0]

    gram = D.t().mm(D)
    L = 2. * torch.linalg.eigvalsh(gram)[-1]
    zero = torch.tensor(0., dtype=X.dtype, device=X.device)
    yt = torch.zeros((n_ex, n_act), dtype=X.dtype, device=X.device)
    if return_history:
        yth = torch.zeros((max_iter, n_ex, n_act), dtype=X.dtype, device=X.device)

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
        yt = yt - grad / L
        yt = torch.max(abs(yt) - lambd / L, zero) * torch.sign(yt)
        if return_history:
            yth[ii] = yt
    if verbose:
        print('inference', ii, seo, sep)
    if return_history:
        return yth[:ii]
    else:
        return yt


def LCA(X, D, lambd, soft=True, max_iter=250, tol=1e-4, eta=None, verbose=False,
        return_history=False):
    n_ex, n_feat = X.shape
    n_act = D.shape[0]

    gram = D.mm(D.t())
    if eta is None:
        L = 2. * torch.linalg.eigvalsh(gram)[-1]
        eta = 1. / L
    u = torch.zeros((n_ex, n_act), dtype=X.dtype, device=X.device)
    s = torch.zeros_like(u)
    if return_history:
        sh = torch.zeros((max_iter, n_ex, n_act), dtype=X.dtype, device=X.device)
    c = gram - torch.eye(n_act, dtype=X.dtype, device=X.device)
    b = X.mm(D.t())
    zero = torch.tensor(0., dtype=X.dtype, device=X.device)

    se = torch.sum((X)**2, dim=1).mean().detach().cpu().numpy()
    seo = se
    any_s = False

    for ii in range(max_iter):
        ci = s.mm(c)
        u = eta * (b - ci) + (1. - eta) * u
        if soft:
            s = torch.sign(u) * torch.max(abs(u) - lambd, zero)
        else:
            s = u * (abs(u) > lambd).type(X.dtype)
        if np.all(np.any(s.detach().numpy(), axis=1)) and not any_s:
            any_s = True
        if return_history:
            sh[ii] = s

        diff = X - s.mm(D)
        sep = torch.sum((diff)**2, dim=1).mean() + lambd * abs(s).sum(dim=1).mean()
        sep = sep.detach().cpu().numpy()
        if (se - sep) / max(1., max(abs(se), abs(sep))) < tol:
            if ii > 0 and any_s:
                break
        se = sep
    if verbose:
        print('inference', ii, seo, sep)
    if return_history:
        return sh[:ii]
    else:
        return s


def cd(Xt, Dt, lambd, max_iter=250, tol=1e-4, verbose=False,
          return_history=False):
    X = Xt.detach().cpu().numpy()
    D = Dt.detach().cpu().numpy()
    n_ex, n_feat = X.shape
    n_act = D.shape[0]

    alpha = lambd / n_feat
    yt = np.zeros((n_ex, n_act))
    if return_history:
        yth = np.zeros((max_iter, n_ex, n_act))
        las0 = Lasso(alpha=alpha, fit_intercept=False, tol=tol, max_iter=max_iter)
        yth = las0.fit(D.T, X.T).coef_
        yth = np.zeros((np.max(las0.n_iter_), n_ex, n_act))
        las = Lasso(alpha=alpha, fit_intercept=False, tol=tol, warm_start=True, max_iter=1)
        for ii in range(np.max(las0.n_iter_)):
            warnings.filterwarnings('ignore')
            yth[ii] = las.fit(D.T, X.T).coef_
        return torch.tensor(yth, dtype=Xt.dtype, device=Xt.device)
    else:
        warnings.filterwarnings('ignore')
        las = Lasso(alpha=alpha, fit_intercept=False, tol=tol, max_iter=max_iter)
        yt = las.fit(D.T, X.T).coef_
        return torch.tensor(yt, dtype=Xt.dtype, device=Xt.device)
