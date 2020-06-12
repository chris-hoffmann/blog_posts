import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from scipy.linalg import inv
from scipy.stats import multivariate_normal 

from .metrics import PAIRWISE_DISTANCE_FUNCS

class GP(): 
    """Gaussian processes (GP) for performing regression.
    
    The GP class provides functionality for:
     - fitting training data
     - predicting numerical labels for unseen test points
     - tuning hyerparameters via maximum likelihood estimation
     - drawing samples from either the prior or posterior distribution
     - as well as several methods for visualization
    
    Parameters
    ----------
    X : array-like
        The inputs of the training set.
    
    y : array-like
        The outputs of the test set.
    
    X_star : 
        The inputs of the test set.
                
    metric : str, default='seuclidean'
        The metric specifies the function for computing pair-wise distances.
        The default keyword returns a function that computes the squared 
        euclidean distance distances.
    
    kernel : str, default='gauss'
        The kernel keyword specifies the kernel subclass. 
        The default keyword returns a Gaussian kernel.

    

    Attributes
    ----------
    X : array-like of shape (samples x features) 
        Array representing inputs of training data.
   
    y_test :
    
    X_star : 
        
    y_test_err :
    
    kernel : 
    
    metric : 
    
    hyper_params : array-like of shape (kernel_params + 1)
    
    """
    
    
    def __init__(self, X, y, X_star=None, noise=0, kernel='gauss', metric='seuclidean'):
        #FIXME add kernel name as property to GP
        self.X = X
        self.y = y
        self.X_star = X_star
        self._noise = noise
        self.kernel = kernel
        self.metric = metric
        self.hyper_params = [noise] + [float(param) for param in self.kernel.kernel_params.values()]
     
    # returns None if attribute is not found  
    def __getattr__(self, item):
        return None
        
    @property
    def noise(self):
        return self._noise

    @noise.setter
    def noise(self, new_noise):
        if new_noise > 0 and isinstance(new_noise, float):
            self._noise = new_noise
        else:
            print("Please enter a valid noise value- positive and float type") 

    @property        
    def mininmization_report(self):
    # Iterate over the keys in dictionary, access value & print line by line
        for key, values in self.opt_report.items():
            print(f'{key}: {values}')
    
    def __repr__(self):
        
        hy_params = {'noise': self.noise}
        hy_params.update(self.kernel.kernel_params)
        
        report = {'model': {'kernel': self.kernel, 'metric': self.metric}, 'hyperparameter': hy_params, \
                      'X_train': self.X, 'y_train': self.y, 'X_test': self.X_star, 'y_test': self._y_test, \
                     'y_test_err': self._y_test_std}

        report_str = "\n".join("{}: {}".format(i,repr(v)) for (i,v) in report.items())
        return f'{report_str}'
    
    
    def draw_from_prior(self, samples=3, plot=False, ax=None):
        """Draws functions from the prior distribution
        
        Parameters
        ----------
        samples : int, default=3
            Hyperparameter vector whose first entry is the noise \sigma_{n}^{2}.
            The other entries repsent kernel parameters.
            If None, log_marginal_likelihood is calculated based
            on ``self.hyperparameters``.
        
        plot : boolean, default=3
        
        Returns
        -------
        log_likelihood : float
            Log-marginal likelihood of hyperparameter alpha for training data.
        
        """

        x = np.linspace(np.min(self.X), np.max(self.X), 101)
        x_mean_vector = np.zeros(len(x))
    
        distance_func = PAIRWISE_DISTANCE_FUNCS[self.metric]
        pair_distances = distance_func(x, x)
        
        x_cov_matrix = self.kernel.calc_kernel_matrix(pair_distances)
        std = np.sqrt(np.diag(x_cov_matrix))
        
        f = multivariate_normal(mean=x_mean_vector, cov=x_cov_matrix, allow_singular=True).rvs(samples)
               
        
        if plot:
            self._plot_gp(mu=x_mean_vector, X=self.X, std=std, sampled_functions=f)
        else:
            f = multivariate_normal(mean=x_mean_vector, cov=x_cov_matrix, allow_singular=True).rvs(samples)
            x = np.tile(x, (samples,1))
            
            return (x.T, f.T)
    
        
    def log_marginal_likelihood(self, alpha=None):
        """Caclulates the logarithm of the marginal likelihood for training
        data as a function of alpha
        
        Parameters
        ----------
        alpha : array-like of shape (n_kernel_params + 1), default=None
            Hyperparameter vector whose first entry is the noise \sigma_{n}^{2}.
            The other entries represent kernel parameters.
            If None, log_marginal_likelihood is calculated based
            on ``self.hyperparameters``.
        
        Returns
        -------
        log_likelihood : float
            Log-marginal likelihood of hyperparameter alpha for training data.
        """
        
        
        pair_distances = self.calc_pair_distances(self.X)
        ## check whether we have already inverted the kernel matrix!!!
    
        if alpha is None:
            K = self.kernel.calc_kernel_matrix(pair_distances)
            K += np.eye(len(pair_distances)) * self.noise
            
        else:
            noise, *kernel_params = alpha
            K = self.kernel.calc_kernel_matrix(pair_distances, kernel_params=kernel_params)
            K += np.eye(len(pair_distances)) * noise
            
        log_marg_likhood_value = -0.5*(np.dot(np.dot(self.y.T, np.linalg.inv(K)), self.y) \
                                      + np.log(np.linalg.det(K)) + len(self.X) * np.log(2*np.pi))
        if alpha is None:
            self._log_marg_likhood_value = log_marg_likhood_value   
        return log_marg_likhood_value
                                                        

        
        
    def tune_hyperparameters(self, constraints=None):
        """Perform hyperparameter optimization by minimizing the negative 
        log marginal likelihood using the BFGS algorithm.
        
        Parameters
        ----------
        constraints : boolean, optional (default=None)
            Constraints for performing constraint optimization.
      
        Returns
        -------
        optima : 
             Most probable hyperparameter vector alpha (local minimum) and 
             the corresponding value of the objective function.
        """
        
        def obj_func(alpha):
            return -self.log_marginal_likelihood(alpha)

        optima = self._minimize_marg_likhood(obj_func, self.hyper_params, constraints)
        return optima
    
    
    
    def predict(self, plot=False, ax=None):
        """Calculates the mean and standard deviation of the posterior
        predictive distribution
        
        Parameters
        ----------
        plot : boolean, optional (default=False)
            
        
        Returns
        -------
        predictive_mean : array-like of shape (test_points)
            Predictive mean of the posterior predictive distribution.
            Contains the predicted outputs of the test points.
        predictive_variance : array-like of shape (test_points x test_points)
            Predictive variance of the posterior predictive distribution.
        
        std_predictive_mean: array-like of shape (test_points)    
            Standard deviation of the predictive mean obtained from the square
            root of the sum of the diagonal elements of the predictive variance
            and the noise variance.
        """
        
        # select function for distance calculations
        distance_func = PAIRWISE_DISTANCE_FUNCS[self.metric]
        
        # compute pair-wise distances
        distances_train = distance_func(self.X, self.X)
        distances_test = distance_func(self.X_star, self.X_star)
        distances_train_test = distance_func(self.X, self.X_star)
        
        # calculate covariance matrices
        K = self.kernel.calc_kernel_matrix(distances_train) + np.eye(len(distances_train)) * self.noise
        K_star = self.kernel.calc_kernel_matrix(distances_train_test)
        K_star_star  = self.kernel.calc_kernel_matrix(distances_test)
        K_inv = inv(K)
        
        predictive_mean = K_star.T.dot(K_inv).dot(self.y)
        predictive_variance = K_star_star - K_star.T.dot(K_inv).dot(K_star)
        
        self._y_test = predictive_mean
        self._y_test_std = np.sqrt(np.diag(predictive_variance) + self.noise)
        self._predictive_var = predictive_variance
        
        if plot:
            self._plot_gp(mu=predictive_mean, X=self.X_star, std=self._y_test_std,\
                          plot_train_data=True, ax=ax)
        else:
            return (predictive_mean.squeeze(), predictive_variance.squeeze())

    
    def draw_from_posterior(self, cov_mat=None, samples=3, plot=False, ax=None):
        """draw functions from the posterior predictive distribution"""
        std = self._y_test_std 
        mu = self._y_test
        cov_mat = self._predictive_var
                
        f_star = multivariate_normal(mean=mu, cov=cov_mat, allow_singular=True).rvs(samples)
        
        min_value = np.min(self.X_star)
        max_value = np.max(self.X_star)         
        x = np.linspace(min_value, max_value, 101)
         
        if plot:
            self._plot_gp(mu=mu, X=self.X_star, std=std, sampled_functions=f_star, plot_train_data=True)
        else:
            x = np.tile(x, (samples,1))
            
            return (x.T, f_star.T)

           
    def _minimize_marg_likhood(self, obj_func, initial_alpha, constraints):
        optimizer = 'BFGS'
        if constraints:
            optimizer = 'COBYLA'
        
        start_values = {'noise': self.noise}
        start_values.update(self.kernel.kernel_params)
        optimal_values = {}
    
   
        opt_res = minimize(obj_func, initial_alpha, constraints=constraints, options={'disp': True})
        if opt_res.success:
            alpha_opt, func_min = opt_res.x, opt_res.fun
            self.minimum_log_marg_likhood = func_min
            self.hyper_params = opt_res.x
            self.noise = np.round(opt_res.x[0], 3)

            # update kernel parameters            
            for opt_value, (key, value) in zip(opt_res.x[1:], self.kernel.kernel_params.items()):
                self.kernel.kernel_params[key] = np.round(opt_value, 3)
        
            optimal_values = {'noise': self.noise}
            optimal_values.update(self.kernel.kernel_params)
        
            opt_report = {'model': {'kernel': self.kernel, 'metric': self.metric}, 'start_values': start_values,\
                      'optimizer': optimizer, 'success': opt_res.success, 'optimal_values': optimal_values}
            self.opt_report = opt_report
        
            return alpha_opt, func_min        

    
    def plot_covariance_matrix(self, ax=None):
        """plot covariance matrix for training instances"""
        if ax is None:
            ax = plt.gca()
        
        # select function for distance calculations
        distance_func = PAIRWISE_DISTANCE_FUNCS[self.metric]
        
        # compute pair-wise distances
        distances_train = distance_func(self.X, self.X)
        K = self.kernel.calc_kernel_matrix(distances_train)
        
        img = ax.imshow(K, cmap="coolwarm", interpolation='none') 
        plt.colorbar(img, ax=ax)
        return(ax)
        
    def _plot_gp(self, mu, X, std, plot_train_data=False, sampled_functions=None, ax=None):
        x = np.linspace(np.min(X), np.max(X), 101)
        
        if ax is None:
            ax = plt.gca()
        
        # plot confidence bands        
        ax.fill_between(x, mu + std, mu - std, alpha=0.1, color='gray', label='68\% confidence band')
        ax.fill_between(x, mu + 1.96 * std, mu - 1.96 * std, alpha=0.1, color='darkgray', label='95\% confidence band')
    
        # plot mean function
        if X is not self.X_star:
            X = x
        ax.plot(X, mu, label='Mean function')
        
        # plot sampled functions
        if sampled_functions is not None:
            for i, sample in enumerate(sampled_functions):
                ax.plot(x, sample, lw=0.5, label=f'Sample {i+1}')
                
        # plot training data        
        if plot_train_data:
            ax.scatter(self.X, self.y, c='r', zorder=3, label=f'Training data')
        ax.legend(loc='upper right', bbox_to_anchor=(1.0, 0.52, 0.5, 0.5))
        
        return(ax)  
            

    def calc_pair_distances(self, X, X_star=None):
        """compute pair-wise distances"""
        distance_func = PAIRWISE_DISTANCE_FUNCS[self.metric]
        if not X_star:
            X_star = X
        return distance_func(X, X_star)