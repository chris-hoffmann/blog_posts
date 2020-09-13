import unittest
import numpy as np

from gp_regression import GP
from gp_regression.kernels import kernel_factory
from gp_regression.metrics import PAIRWISE_DISTANCE_FUNCS

from scipy.optimize import LinearConstraint
from numpy.testing import assert_


class Test_GP_Class(unittest.TestCase):
    
    def setUp(self):
        np.random.seed(42)
        X = np.array([-5,-3,0,0.1,1,4.9,5])
        y = np.array([0,-0.5,1,0.7,0,1,0.7])
        X_star = np.linspace(-6, 6, 101)
        
        gauss_kernel_large_var = kernel_factory('gaussian', length_scale=3)
        self.gpr_model_large_var = GP(X, y, X_star, metric='seuclidean', kernel=gauss_kernel_large_var, noise=0.2)
   
   
    def test_calc_kernel_matrix(self): 
        K_check = np.load('./test_data/K_check.npy')
        
        distance_func = PAIRWISE_DISTANCE_FUNCS[self.gpr_model_large_var.metric]
        pair_distances_X = distance_func(self.gpr_model_large_var.X, self.gpr_model_large_var.X)
        actual = self.gpr_model_large_var.kernel.calc_kernel_matrix(pair_distances_X)
        
        np.testing.assert_allclose(actual, K_check, err_msg='calculated kernel matrix not as expected')  
    
    
    def test_draw_from_prior(self):  
        f_check = np.load('./test_data/f_check.npy')
        
        _, f_drawn = self.gpr_model_large_var.draw_from_prior(plot=False)
        
        np.testing.assert_allclose(f_drawn, f_check, err_msg='drawn prior functions not as expected') 
    
    
    def test_calc_log_marginal_likelihood(self):    
        self.assertEqual(round(self.gpr_model_large_var.calc_log_marginal_likelihood(), 4), -7.5071, 'value of log marginal likelihood incorrect')
               
       
    def test_tune_hyperparameters(self):
        alpha_check = np.array([ 0.22233858, 10.66407922])
        
        alpha, minimum = self.gpr_model_large_var.tune_hyperparameters()
        
        np.testing.assert_allclose(alpha, alpha_check, err_msg='found hyperparameters not as expected') 
        
        
    def test_predict(self):  
        pred_mean_check = np.load('./test_data/pred_mean.npy')
    
        pred_mean,_ = self.gpr_model_large_var.predict(plot=False)
        
        np.testing.assert_allclose(pred_mean, pred_mean_check, err_msg='predictive mean not as expected') 
        
        
    def test_draw_from_posterior(self):   
        f_star_check = np.load('./test_data/f_star_check.npy')
        
        _ = self.gpr_model_large_var.predict(plot=False)
        _, f_star_drawn = self.gpr_model_large_var.draw_from_posterior(plot=False)
       
        np.testing.assert_allclose(f_star_drawn, f_star_check, err_msg='drawn posterior functions not as expected')