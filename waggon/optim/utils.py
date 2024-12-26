import os
import numpy as np


_PRIMES = np.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97])

def _get_olhs_num(n):
    '''
    Private function to select the number of sampling points for orthogonal Latin hypercube sampling.
    '''
    return _PRIMES[_PRIMES ** 2 > n]**2

def create_dir(func, acqf_name, surr_name, base_dir='test_results'):
    
    res_path = base_dir
    if not os.path.isdir(res_path):
        os.mkdir(res_path)
    
    func_dir = func.name if func.dim == 2 else func.name + str(func.dim)

    res_path += f'/{func_dir}'
    if not os.path.isdir(res_path):
        os.mkdir(res_path)
    
    res_path += f'/{acqf_name}'
    if not os.path.isdir(res_path):
        os.mkdir(res_path)
    
    res_path += f'/{surr_name}'
    if not os.path.isdir(res_path):
        os.mkdir(res_path)
    
    return res_path


class PyXABFunctionWrapper:
    def __init__(self, func):
        """
        Wrapper over waggon.functions.Function objects for multi-armed bandits
        optimization using PyXAB library.
        """
        self.func = func
    
    def __call__(self, x: list):
        x_t = np.expand_dims(x, 0)
        reward = -self.func(x_t)
        
        # A crutch, will be removed in future
        if reward.size > 1:
            raise ValueError("Reward must be 1d-array!")

        return reward.flatten()[0]
