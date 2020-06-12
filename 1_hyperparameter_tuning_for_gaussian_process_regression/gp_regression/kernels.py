"""Kernels for Gaussian process regression. 
Implementation is based on the Factory design pattern.
"""

import numpy as np
from abc import ABC, abstractmethod

def kernel_factory(kernel_type, **kwargs):
    "Factory method for selecting a Kernel Subclass"
    kernel_dict = {'gaussian': Gaussian, 'laplacian':Laplacian}
    kernel = kernel_dict[kernel_type] 
    kernel_instance = kernel(kernel_type, **kwargs)
    return kernel_instance

class Kernel(ABC):
    """Base class for all kernels."""
    @abstractmethod
    def calc_kernel_matrix(self, pair_distances, kernel_params=None):
        """Compute the kernel matrix."""
        pass
    
    def __repr__(self):
        pass


class Gaussian(Kernel):
    """Isotropic Gaussian kernel.
    
    Parameters
    ----------
    l : float, default=None
        The variance or length-scale of the kernel.
    """
    
    def __init__(self, name, variance=1.):
        self.kernel_params = {'variance': variance}
    
    def calc_kernel_matrix(self, pair_distances, kernel_params=None):
        if kernel_params:
            variance = kernel_params[0]
        else: 
            variance = self.kernel_params['variance']
        return np.exp(-pair_distances/(2 * variance**2))

    def __repr__(self):
        return f'{self.__class__.__name__}'
    
    def plot_covar_function(self, metric=None, ax=None):
        """plots the covariance function as function of the distance between
        points in the interval [0,4] given a distance metric"""
        if ax is None:
            ax = plt.gca()
        
        x = np.linspace(0,4)
        if metric is None:
            metric = 'seuclidean'
        distance_func = PAIRWISE_DISTANCE_FUNCS[metric]
        pair_distances = distance_func(x,x)[0,:]
    
        f = self.calc_kernel_matrix(pair_distances)
        ax.plot(x,f)
        return(ax)

    
class Laplacian(Kernel):
    """Isotropic Laplace kernels.

    Parameters
    ----------
    l : float, default=1.
        The variance or length-scale of the kernel.
    """
    def __init__(self, name, variance=1.):
        self.kernel_params = {'variance': variance}

    def calc_kernel_matrix(self, pair_distances, kernel_params=None):
        if kernel_params:
            variance = kernel_params[0]
        else: 
            variance = self.kernel_params['variance']
        return np.exp(-pair_distances/variance)

    def __repr__(self):
        return f'{self.__class__.__name__}'