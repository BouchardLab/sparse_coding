import numpy as np

import torch
from torch import optim

from scipy.optimize import minimize

from .inference import infer


def learn(learn_method, infer_method, X, D, lambd, *args):
    """Run sparse coding learning. This function dispatches different methods.

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

    if learn_method.lower() == 'bfgs':
        D = bfgs(infer_method, X, D, lambd, *args)
    elif learn_method.lower() == 'optim':
        D = torchoptim(infer_method, X, D, lambd, *args)
    else:
        raise ValueError
    return D / np.linalg.norm(D, axis=1, keepdims=True)


def f(infer_method, D, X, lambd, verbose=False):
    D_norm = torch.norm(D, dim=1, keepdim=True)
    D = D / D_norm
    A = infer(infer_method, X, D.detach(), lambd, verbose=verbose)
    cost = torch.sum((X - A.mm(D))**2, dim=1).mean()
    if verbose:
        print('learn', cost.detach().cpu().numpy())
    return cost


def f_df(D_flat, infer_method, shape, Xt, lambd):
    D_flatt = torch.tensor(D_flat, requires_grad=True, dtype=Xt.dtype, device=Xt.device)
    Dto = D_flatt.reshape(shape)
    Dto_norm = torch.norm(Dto, dim=1, keepdim=True)
    Dt = Dto / Dto_norm
    At = infer(infer_method, Xt, Dt.detach(), lambd)
    cost = torch.sum((Xt - At.mm(Dt))**2, dim=1).mean()
    cost.backward()
    grad = D_flatt.grad
    return cost.detach().cpu().numpy().astype(float), grad.detach().cpu().numpy().astype(float)


def callback(D_flat, infer_method, shape, Xt, lambd):
    D_flatt = torch.tensor(D_flat, dtype=Xt.dtype, device=Xt.device)
    Dto = D_flatt.reshape(shape)
    Dto_norm = torch.norm(Dto, dim=1, keepdim=True)
    Dt = Dto / Dto_norm
    At = infer(infer_method, Xt, Dt, lambd)
    cost = torch.sum((Xt - At.mm(Dt))**2, dim=1).mean()
    print('learn', cost.detach().cpu().numpy())


def bfgs(infer_method, X, D, lambd, verbose=False):
    args = (infer_method, D.shape, X, lambd)
    Dnp = D.ravel()
    _callback = None
    if verbose:
        _callback = lambda params: callback(params, *args)
        callback(Dnp, *args)
    opt = minimize(f_df, Dnp, args=args, jac=True, method='L-BFGS-B',
                   callback=_callback)
    return opt.x.reshape(*D.shape)


def torchoptim(infer_method, Xt, D, lambd, batch_size=256, max_epochs=10,
               patience_batches=None, learn_tol=1e-4):
    Dt = torch.tensor(D, requires_grad=True, device=Xt.device, dtype=Xt.dtype)
    opt = optim.Adam([Dt])
    batches = np.array(torch.split(Xt, batch_size), dtype=object)
    if patience_batches is None:
        patience_batches = 5 * batches.size
    track_loss = None
    lowest = np.finfo(float).max
    frac = (patience_batches - 1.) / patience_batches
    patience = 0
    for ep in range(max_epochs):
        batches = np.random.permutation(batches)
        for ii, b in enumerate(batches):
            if (ii % 20) == 0:
                verbose = True
            else:
                verbose = False
            opt.zero_grad()
            loss = f(infer_method, Dt, b, lambd, verbose)
            loss.backward()
            opt.step()
            loss = loss.detach().cpu().numpy()
            if track_loss is None:
                track_loss = loss
            else:
                track_loss = frac * track_loss + (1. - frac) * loss
                if (lowest - track_loss) / np.max([1., abs(track_loss), abs(lowest)]) < learn_tol:
                    patience += 1
                else:
                    patience = 0
                lowest = min(track_loss, lowest)
            print(ii, patience, loss, track_loss)
            if patience >= patience_batches:
                break
        if patience >= patience_batches:
            break
    return Dt.detach().cpu().numpy()
