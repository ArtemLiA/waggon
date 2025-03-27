import time
import pickle

import numpy as np
from tqdm import tqdm

from PyXAB.algos.Zooming import Zooming
from PyXAB.algos.HOO import T_HOO
from PyXAB.algos.DOO import DOO
from PyXAB.algos.SOO import SOO
from PyXAB.algos.StoSOO import StoSOO
from PyXAB.algos.HCT import HCT
from PyXAB.algos.POO import POO
from PyXAB.algos.GPO import GPO
from PyXAB.algos.PCT import PCT
from PyXAB.algos.SequOOL import SequOOL
from PyXAB.algos.StroquOOL import StroquOOL
from PyXAB.algos.VROOM import VROOM
from PyXAB.algos.VHCT import VHCT
from PyXAB.algos.VPCT import VPCT

from PyXAB.partition.BinaryPartition import BinaryPartition
from PyXAB.partition.RandomBinaryPartition import RandomBinaryPartition
from PyXAB.partition.DimensionBinaryPartition import DimensionBinaryPartition
from PyXAB.partition.KaryPartition import KaryPartition
from PyXAB.partition.RandomKaryPartition import RandomKaryPartition

from .base import Optimiser
from .utils import create_dir
from .utils import PyXABFunctionWrapper

ALGORITHMS = {
    "Zooming": Zooming,
    "T-HOO": T_HOO,
    "DOO": DOO,
    "SOO": SOO,
    "StoSOO": StoSOO,
    "HCT": HCT,
    "POO": POO,
    "GPO": GPO,
    "PCT": PCT, 
    "SequOOL": SequOOL,
    "StroquOOL": StroquOOL,
    "VROOM": VROOM,
    "VHCT": VHCT,
    "VPCT": VPCT, 
}


PARTITIONS = {
    "binary": BinaryPartition,
    "rand-binary": RandomBinaryPartition,
    "dim-binary": DimensionBinaryPartition,
    "kary": KaryPartition,
    "rand-kary": RandomKaryPartition,
}


class PyXABOptimizer(Optimiser):
    def __init__(self, func, algo="T-HOO", partition='binary', **kwargs):
        """
        Multi-armed bandit optimizer. Uses algorithms from PyXAB library.

        """
        super(PyXABOptimizer).__init__()
        self.func         = func
        self.func_wrapper = PyXABFunctionWrapper(func)
        self.max_iter     = kwargs.pop('max_iter', 1000)
        self.eps          = kwargs.pop('eps', 1e-1)
        self.error_type   = kwargs.pop('error_type', 'x')
        self.verbose      = kwargs.pop('verbose', 1)
        self.save_results = kwargs.pop('save_results', True)

        kwargs["partition"] = PARTITIONS[partition]
        kwargs["domain"] = self.func.domain.tolist()

        algo_creator = ALGORITHMS[algo]
        self.algo = algo_creator(**kwargs)

        # Experiments parameters
        self.errors = []
        self.res = None
        self.params = None

    def predict(self, time: float) -> np.ndarray:
        """
        Add information about value at point of interest and return best candidate.
        """
        point = self.algo.pull(time)
        reward = self.func_wrapper(point)
        self.algo.receive_reward(time, reward)

        best_point = self.algo.get_last_point()
        return np.array(best_point)
    
    def compute_error(self, x_best, y_best):
        x_glob_min = self.func.glob_min
        y_glob_min = self.func(np.expand_dims(x_glob_min, 0)).flatten()

        if x_glob_min.ndim == 1:
            x_glob_min = x_glob_min[np.newaxis]

        if self.error_type == 'x':
            return np.linalg.norm(x_glob_min - x_best, axis=1).min()

        y_glob_min = self.func(x_glob_min)

        if y_glob_min.ndim == 2:
            y_glob_min = y_glob_min[0]
        
        if self.func.log_transform:
            y_best = np.exp(y_best) + self.func.log_eps
            y_glob_min = np.exp(y_glob_min) + self.func.log_eps

        if self.error_type == 'f':
            return np.linalg.norm(y_glob_min - y_best)

        raise ValueError(f"Unsupported error type: {self.error_type}")
    
    def evaluate(self, x_best) -> float:
        y_best = self.func(np.expand_dims(x_best, 0)).flatten()

        error = self.compute_error(x_best, y_best)
        self.errors.append(error)
        
        # Log best point
        if self.params is None:
            dim = self.func.dim
            self.params = np.zeros((0, dim))
            self.res = np.zeros((0, 1))
    
        self.params = np.concatenate((
            self.params, np.expand_dims(x_best, 0)
        ))
        self.res = np.concatenate((
            self.res, np.expand_dims(y_best, 0)
        ))

        return error
    
    def optimise(self, X=None, y=None, **kwargs) -> None:
        if self.verbose == 0:
            opt_loop = range(self.max_iter)
        else:
            opt_loop = tqdm(
                range(self.max_iter),
                desc='Optimization loop started...',
                leave=True,
                position=0
            )
        
        for t in opt_loop:
            x_best = self.predict(t)
            error = self.evaluate(x_best)

            if error <= self.eps:
                print('Experiment finished successfully')
                break
        
        if error > self.eps:
            print('Experiment failed')
        
        if self.save_results:
            self._save()
