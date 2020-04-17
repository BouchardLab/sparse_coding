import numpy as np
import torch

from .inference import infer
from .learning import learn


class SparseCoding(object):
    """Learning and inference for a sparse coding model.

    Parameters
    ----------
    n_sources : int
        Number of latent variables (sources) in the model.
    lambd : float
        L1 penalty weight.
    infer_methods: str
        Method for infering sparse codes. Used during learning and by `transform`.
    learn_method: str
        Method used for dictionary learning.
    """
    def __init__(self, n_sources, lambd=.1, infer_method='fista', learn_method='optim',
                 verbose=False, seed=20200412, device='cpu', dtype=torch.float64, **kwargs):
        self.n_sources = n_sources
        self.lambd = lambd
        self.infer_method = infer_method
        self.learn_method = learn_method
        self.verbose = verbose
        self.rng = np.random.RandomState(seed)
        self.device = device
        self.dtype = dtype
        self.kwargs = kwargs
        self.D = None

    def fit(self, X):
        """Fit a sparse coding model.

        Parameters
        ----------
        X : ndarray (examples, features)
            Training data.
        """
        if not torch.is_tensor(X):
            X = torch.tensor(X, dtype=self.dtype, device=self.device)
        if self.D is None:
            self.D = self.rng.randn(self.n_sources, X.shape[1])
            self.D /= np.linalg.norm(self.D, axis=1, keepdims=True)
        self.D = learn(self.learn_method, self.infer_method, X, self.D, self.lambd,
                       verbose=self.verbose, **self.kwargs)
        return self

    def transform(self, X, return_history=False):
        """Infer the sparse codes for given data.

        Parameters
        ----------
        X : ndarray (examples, features)
            Data to infer sparse codes for.
        """
        X = torch.tensor(X, dtype=self.dtype, device=self.device)
        D = torch.tensor(self.D, dtype=self.dtype, device=self.device)
        return infer(self.infer_method, X, D, self.lambd, verbose=self.verbose,
                     return_history=return_history, **self.kwargs).detach().cpu().numpy()
