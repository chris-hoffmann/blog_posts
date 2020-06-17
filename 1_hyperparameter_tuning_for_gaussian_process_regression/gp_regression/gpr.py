import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from scipy.linalg import inv
from scipy.stats import multivariate_normal 

from .metrics import PAIRWISE_DISTANCE_FUNCS

np.random.seed(42)

class GP(): 
    """Gaussian processes (GP) for performing regression.
    
    The GP class provides functionality for:
     - fitting training data
     - predicting numerical labels for unseen test points
     - tuning hyerparameters via a maximum likelihood estimation
     - drawing samples from either the prior or posterior distribution
     - as well as several methods for visualization
    
    Parameters
    ----------
    X : array-like of shape (n_training_instances x x_dim)
        The inputs of the training set.
    
    y : array-like of shape (n_training_instances x y_dim)
        The outputs of the test set.
    
    X_star : array-like of shape (n_test_instances x x_dim), optional (default=None)
        The inputs of the test set.
    
    noise : float, optional (default=0.)
        The noise level $\sigman_n^2$.
    
    metric : str, default='seuclidean'
        The metric specifies the function for computing pair-wise distances.
        The default keyword returns a function that computes the squared 
        euclidean distance distances.
    
    kernel : str, default='gauss'
        The kernel keyword specifies the Kernel Subclass. 
        The default keyword returns a Gaussian kernel.

    
    Attributes
    ----------
    X : array-like of shape (n_samples x n_features) 
        Array representing inputs of training data.
   
    y : array-like of shape (n_samples x output_dim) 
        Vector or array representing outputs of training data.
    
    X_star : array-like of shape (n_samples x n_features), default=None 
        Array representing inputs of test data.
        
    y_test_err :
    
    y_test_std :
    
    kernel : kernel object
    
    metric : str
    
    value_log_marg_likhood
    
    """
    
    
    def __init__(self, X, y, X_star=None, noise=0., kernel='gauss', metric='seuclidean'):
        #FIXME add kernel name as property to GP
        self.X = X
        self.y = y
        self.X_star = X_star
        self._noise = noise
        self.kernel = kernel
        self.metric = metric
    
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
    def hyper_params(self):
        self._hyper_params = [self._noise] + [float(param) for param in self.kernel.kernel_params.values()]
        return self._hyper_params
    
    @property        
    def mininmization_report(self):
        for key, values in self.opt_report.items():
            print(f'{key}: {values}')
    
    def __repr__(self):   
        hy_params = {'noise': self.hyper_params[0]}
        hy_params.update(self.kernel.kernel_params)
        
#        {name: value for (name, value) in zip(list(self.kernel.kernel_params.keys()), gpr_model_small_var.hyper_params[1:])}
        
        report = {'model': {'kernel': self.kernel, 'metric': self.metric}, 'hyperparameter': hy_params, \
                      'X_train': self.X, 'y_train': self.y, 'X_test': self.X_star, 'y_test': self._y_test, \
                     'y_test_err': self._y_test_std}

        report_str = "\n".join("{}: {}".format(i,repr(v)) for (i,v) in report.items())
        return f'{report_str}'
    
    
    def draw_from_prior(self, samples=3, plot=False, ax=None):
        """Draws functions from the prior distribution.
        
        Parameters
        ----------
        samples : int, default=3
            Number of sampled functions.
        
        plot : boolean, optional (default=False)
            If True the sampled functions are plotted.
            
        ax : boolean, optional (default=None)
            Instance of Matplotlib axes. Specifies the axes to draw the plot onto.
            If None, the current axes is used.
            
        Returns
        -------
        inputs of sampled_functions : array-like of shape (101 x n_samples x x_dim)
            Returns the input vector x that contains 101 values ranged between the
            minimum and maximum of the training inputs.
        
        outputs of samples functions : array-like of shape (101 x n_samples x y_dim)
            Returns the output values f(x) of the sampled functions.
        """

        x = np.linspace(np.min(self.X), np.max(self.X), 101)
        x_mean_vector = np.zeros(len(x))
    
        pair_distances = self.calc_pair_distances(x)
        x_cov_matrix = self.kernel.calc_kernel_matrix(pair_distances)
        std = np.sqrt(np.diag(x_cov_matrix))
        
        f = multivariate_normal(mean=x_mean_vector, cov=x_cov_matrix, allow_singular=True).rvs(samples)          
        
        if plot:
            self._plot_gp(mu=x_mean_vector, X=self.X, std=std, sampled_functions=f, ax=ax)
        else:
            x = np.tile(x, (samples,1))      
            return (x.T, f.T)
    
        
    def log_marginal_likelihood(self, alpha=None):
        """Calculates the logarithm of the marginal likelihood for training
        data as a function of the hyperparameter vector alpha.
        
        Parameters
        ----------
        alpha : array-like of shape (n_kernel_params + 1), optional (default=None)
            Hyperparameter vector whose first entry is the noise \sigma_{n}^{2}.
            The other entries represent kernel parameters.
            If None, the log_marginal_likelihood is calculated based
            on ``self.hyperparameters``.
        
        Returns
        -------
        log_likelihood : float
            Log-marginal likelihood of hyperparameter alpha for training data.
        """
        
        pair_distances = self.calc_pair_distances(self.X)
    
        # Compute the covariance matrix.
        # If alpha is None, use the default values for the hyperparameters.
        if alpha is None:
            K = self.kernel.calc_kernel_matrix(pair_distances)
            K += np.eye(len(pair_distances)) * self.noise            
        else:
            noise, *kernel_params = alpha
            K = self.kernel.calc_kernel_matrix(pair_distances, kernel_params=kernel_params)
            K += np.eye(len(pair_distances)) * noise
        
        # Calculate the log marginal likelihood
        log_marg_likhood_value = -0.5*(np.dot(np.dot(self.y.T, np.linalg.inv(K)), self.y) \
                                      + np.log(np.linalg.det(K)) + len(self.X) * np.log(2*np.pi))
        
        if alpha is None:
            self._log_marg_likhood_value = log_marg_likhood_value   
        
        return log_marg_likhood_value
                                                        
        
    def tune_hyperparameters(self, constraints=None):
        """Perform hyperparameter optimization by minimizing the negative 
        logarithm of the marginal likelihood using the BFGS algorithm.
        
        Parameters
        ----------
        constraints : boolean, optional (default=None)
            Constraints for performing constraint optimization with the COBYLA
            algorithm.
      
        Returns
        -------
        minimization_report : dict
            Summarizes the attempted minimization.
        
        optimal_hyperparameter : array-like of shape (n_kernel_params + 1)
            If the minimizaton terminates successfully, it return the optimal
            hyperparameter vector that corresponds to a local minimum of the 
            negative log marginal likelihood, which serves as our objective function.
             
        local_minimum : float    
             Returns the local mininum of the log marginal likelihood if the 
             minimizaton terminates successfully.
        """
        
        def obj_func(alpha):
            return -self.log_marginal_likelihood(alpha)

        optima = self._minimize_marg_likhood(obj_func, self.hyper_params, constraints)
        return optima
    
    
    
    def predict(self, plot=False, ax=None):
        """Calculates the mean and standard deviation of the posterior
        predictive distribution.
        
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

        # calculate pair-distances
        distances_train = self.calc_pair_distances(self.X)
        distances_test = self.calc_pair_distances(self.X_star)
        distances_train_test = self.calc_pair_distances(self.X, self.X_star)
    
        # calculate covariance matrices
        K = self.kernel.calc_kernel_matrix(distances_train) + np.eye(len(distances_train)) * self.noise
        K_star = self.kernel.calc_kernel_matrix(distances_train_test)
        K_star_star  = self.kernel.calc_kernel_matrix(distances_test)
        K_inv = inv(K)
        
        # compute posterior predictive
        predictive_mean = K_star.T.dot(K_inv).dot(self.y)
        predictive_variance = K_star_star - K_star.T.dot(K_inv).dot(K_star)
        
        # Update attributes
        self._y_test = predictive_mean
        self._y_test_std = np.sqrt(np.diag(predictive_variance) + self.noise)
        self._predictive_var = predictive_variance # needed for sampling f_star
        
        if plot:
            self._plot_gp(mu=predictive_mean, X=self.X_star, std=self._y_test_std,\
                          plot_train_data=True, ax=ax)
        else:
            return (predictive_mean.squeeze(), predictive_variance.squeeze())

    
    def draw_from_posterior(self, samples=3, plot=False, ax=None):  
        """Draw functions from the posterior predictive if the distribution is available.        
        
        Parameters
        ----------
        samples : int, default=3
            Number of sampled functions.
        
        plot : boolean, optional (default=False)
            If True the sampled functions are plotted.
            
        ax : boolean, optional (default=None)
            Instance of Matplotlib axes. Specifies the axes to draw the plot onto.
            If None, the current axes is used.
        
        Returns
        -------
        inputs of sampled_functions : array-like of shape (101 x n_samples x x_dim)
            Returns the input vector x that contains 101 values ranged between the
            minimum and maximum of the training inputs.
        
        outputs of sampled functions : array-like of shape (101 x n_samples x y_dim)
            Returns the output values f_{*}(x) of the sampled functions.
        
        """
        
        # Check, whether the predictive posterior distribution is available:
        assert (self._y_test is not None), "Cannot draw from the predictive posterior, the distribution is unkown."
        
        
        mu = self._y_test
        std = self._y_test_std 
        cov_mat = self._predictive_var
                
        x = np.linspace(np.min(self.X_star), np.max(self.X_star), 101)            
        f_star = multivariate_normal(mean=mu, cov=cov_mat, allow_singular=True).rvs(samples)
             
        if plot:
            self._plot_gp(mu=mu, X=self.X_star, std=std, sampled_functions=f_star, plot_train_data=True)
        else:
            x = np.tile(x, (samples,1))
            return (x.T, f_star.T)

           
    def _minimize_marg_likhood(self, obj_func, initial_alpha, constraints):
        """Performs the minimization of the objective function."""
        
        optimizer = 'BFGS'
        if constraints:
            optimizer = 'COBYLA'
        
        # prepare minimization report
        start_values = {'noise': self.noise}
        start_values.update(self.kernel.kernel_params)
        
        optimal_values = {}
        
        opt_report = {'model': {'kernel': self.kernel, 'metric': self.metric}, 'start_values': start_values,\
                      'optimizer': optimizer, 'success': False}
    
        # perform minimization 
        opt_res = minimize(obj_func, initial_alpha, constraints=constraints, options={'disp': True})
        
        if opt_res.success:      
            alpha_opt, func_min = opt_res.x, opt_res.fun
            self._log_marg_likhood_value = func_min
            self._hyper_params = opt_res.x
            self.noise = np.round(opt_res.x[0], 3)

            # update kernel parameters            
            for opt_value, (key, value) in zip(opt_res.x[1:], self.kernel.kernel_params.items()):
                self.kernel.kernel_params[key] = np.round(opt_value, 3)
        
            optimal_values = {'noise': self.noise}
            optimal_values.update(self.kernel.kernel_params)
            
            opt_report['success'] = opt_res.success
            
#             opt_report = {'model': {'kernel': self.kernel, 'metric': self.metric}, 'start_values': start_values,\
#                       'optimizer': optimizer, 'success': opt_res.success, 'optimal_values': optimal_values}
            
            opt_report.update({'optimal_values': optimal_values})
            self.opt_report = opt_report
            return alpha_opt, func_min 
#            
        self.opt_report = opt_report 
        
    
    def plot_covariance_matrix(self, ax=None):
        """Plot the noise-free covariance matrix for training instances."""
        if ax is None:
            ax = plt.gca()
        
        distances_train = self.calc_pair_distances(self.X)
        K = self.kernel.calc_kernel_matrix(distances_train)

        img = ax.imshow(K, cmap="coolwarm", interpolation='none') 
        plt.colorbar(img, ax=ax)
        return(ax)

    
    def _plot_gp(self, mu, X, std, sampled_functions=None, plot_train_data=False, ax=None):
        """Plot predictions and/or sampled functions."""
        
        # The lower-case x defines the interval in which the sampled functions are plotted.
        x = np.linspace(np.min(X), np.max(X), 101)
         
        # The upper-case X refers to the relevant input set (X_train for the prior, X_test for the posterior)
        # and defines the interval in which the mean function is plotted.
        # For X = X_train this interval is equal to x.             
            
        if ax is None:
            ax = plt.gca()
        
        # plot 1: mean function
        if X is not self.X_star:
            X = x
        ax.plot(X, mu, label='Mean function')
            
        # plot 2: confidence bands        
        ax.fill_between(X, mu + std, mu - std, alpha=0.2, color='gray', label='68\% confidence band')
        ax.fill_between(X, mu + 1.96 * std, mu - 1.96 * std, alpha=0.2, color='darkgray', label='95\% confidence band')
    
        # plot 3: sampled functions (optional)
        if sampled_functions is not None:
            for i, sample in enumerate(sampled_functions):
                ax.plot(x, sample, lw=0.5, label=f'Sample {i+1}')
                
        # plot 4: training data (optional)      
        if plot_train_data:
            ax.scatter(self.X, self.y, c='r', zorder=3, label=f'Training data')
        ax.legend(loc='upper right', bbox_to_anchor=(1.0, 0.52, 0.5, 0.5))
        
        return(ax)             

    def calc_pair_distances(self, X, X_star=None):
        """Calculate pair-wise distances between all instances."""
        distance_func = PAIRWISE_DISTANCE_FUNCS[self.metric]
        if X_star is None:
            X_star = X
        return distance_func(X, X_star)