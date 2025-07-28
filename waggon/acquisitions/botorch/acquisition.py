"""
Acquisition functions which implementation is based on BoTorch
"""
from typing import Optional

import numpy as np
import torch
import botorch.acquisition as bacq
import botorch.fit as bfit

from ..base import Acquisition


class EI(Acquisition):
    def __init__(self, model, best_f: Optional[float] = None):
        self._model = model
        
        if best_f is None:
            self._best_f = model.model.train_targets.min().item()
        else:
            self._best_f = best_f

        self._acqf = bacq.ExpectedImprovement(
            model=self._model,
            best_f=self._best_f,
            maximize=False
        )

    def __call__(self, X: np.ndarray):
        X_data = torch.from_numpy(X).double().unsqueeze(1)
        return self._acqf(X_data).detach().numpy()


class LogEI(Acquisition):
    def __init__(self, model, best_f: Optional[float] = None):
        self._model = model
        
        if best_f is None:
            self._best_f = model.model.train_targets.min().item()
        else:
            self._best_f = best_f

        self._acqf = bacq.LogExpectedImprovement(
            model=self._model,
            best_f=self._best_f,
            maximize=False
        )

    def __call__(self, model, X: np.ndarray):
        X_data = torch.from_numpy(X).double().unsqueeze(1)
        return self._acqf(X_data).detach().numpy()


class PI(Acquisition):
    def __init__(self, model, best_f: Optional[float] = None):
        self._model = model
        
        if best_f is None:
            self._best_f = model.model.train_targets.min().item()
        else:
            self._best_f = best_f

        self._acqf = bacq.ProbabilityOfImprovement(
            model=self._model,
            best_f=self._best_f,
            maximize=False
        )
    
    def __call__(self, X: np.ndarray):
        X_data = torch.from_numpy(X).double().unsqueeze(1)
        return self._acqf(X_data).detach().numpy()


class LCB(Acquisition):
    def __init__(self, model, beta: float):
        self._model = model
        self._beta = beta
        self._acqf = bacq.UpperConfidenceBound(
            model=self._model,
            beta=self._beta
        )

    def __call__(self, X: np.ndarray) -> np.ndarray:
        X_data = torch.from_numpy(X).double().unsqueeze(1)
        return self._acqf(X_data).detach().numpy()
