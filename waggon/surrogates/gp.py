from typing import Tuple
import warnings

import GPy # TODO: remove it
import numpy as np

import torch

import botorch.models as bmodels
import botorch.fit as bfit
import gpytorch.mlls as gmlls

from .base import Surrogate


class GaussianProcess(Surrogate):
    def __init__(self):
        self._model = None
        self._mll = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if y.ndim == 1:
            y = np.expand_dims(y, 1)

        train_X = torch.from_numpy(X).double()
        train_Y = torch.from_numpy(y).double()

        self._model = bmodels.SingleTaskGP(
            train_X=train_X,
            train_Y=train_Y
        )
        self._mll = gmlls.ExactMarginalLogLikelihood(
            likelihood=self._model.likelihood,
            model=self._model
        )
        bfit.fit_gpytorch_mll(self._mll)

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self._model is None:
            raise ValueError("Model is not fitted!")

        X_data = torch.from_numpy(X).double()
        mult_norm = self._model(X_data) # gpytorch.distibutions.MultivariateNormal
        
        mu = mult_norm.loc.detach().numpy()
        std = mult_norm.stddev.detach().numpy()

        return mu, std
    
    @property
    def model(self) -> bmodels.SingleTaskGP | None:
        if self._model is None:
            warnings.warn(
                "BoTorch model of GP is not initialized!",
                category=UserWarning
            )

        return self._model
    

class GP(Surrogate):
    def __init__(self, **kwargs):
        super(GP, self).__init__()

        self.name     = 'GP'
        self.model    = kwargs['model'] if 'model' in kwargs else None
        self.kernel   = kwargs['kernel'] if 'kernel' in kwargs else None
        self.mean     = kwargs['mean'] if 'mean' in kwargs else None
        self.verbose  = kwargs['verbose'] if 'verbose' in kwargs else 1
    
    def fit(self, X, y):

        if self.model is None:
            
            if self.kernel is None:
                self.kernel = GPy.kern.Matern32(input_dim=X.shape[-1], lengthscale=1.0)

            self.model = GPy.models.GPRegression(X.astype(np.float128), y.astype(np.float128),
                                                 kernel = self.kernel,
                                                 mean_function = self.mean)
        self.mu, self.std = np.mean(y), np.std(y)
        y = (y - self.mu) / (self.std + 1e-8)
        
        self.model.set_XY(X=X.astype(np.float128) , Y=y.astype(np.float128))
        
        self.model.optimize(optimizer='lbfgsb')

    def predict(self, X):

        f, var = self.model.predict(X.astype(np.float128))
        std = np.sqrt(var)

        f += self.mu
        std *= self.std

        return f.astype(np.float64), std.astype(np.float64)
    