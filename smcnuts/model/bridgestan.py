import os
import json
import bridgestan as bs
import autograd.numpy as np


class StanModel():
    """
    Class for handling Stan models. Primarily used to calculate the logpdf and gradient of the logpdf of a Stan model.
    
    """

    def __init__(self, model_name, model_path, data_path=None):
        self.model_name = model_name
        self.model_path = model_path
        self.data_path = data_path

        self.stan_model = bs.StanModel.from_stan_file(model_path, data_path if data_path else None)

        if self.data_path and data_path:
            self._update_phi(1.0)

        self.dim = self.stan_model.param_unc_num()
        self.constrained_dim = self.stan_model.param_num(include_tp=True, include_gq=True)
        self.param_names = self.stan_model.param_names()
        self.last_phi = 1.0

    def logpdf(self, x, phi=1.0, adjust_transform=True):
        """
        Calculate the log density of the target distribution at x

        Args:
            x: Unconstrained parameter values to evaluate the log density at
            phi: The temperature of the target distribution, default is no tempering, i.e. phi = 1.0
        
        Returns:
            The log density of the target distribution at x (or an array of log densities if x is a 2D array of samples)
        """

        if self.data_path and phi != self.last_phi:
            self._update_phi(phi)

        # If x is a 1D array, calculate the logpdf of x
        if x.ndim == 1:
            try:
                return self.stan_model.log_density(x)
            except:
                # Return negative infinity in log space if the above fails.
                return -np.inf
        else:
            N = x.shape[0]
            p_logpdf_x_new = np.zeros(N)
            for i in range(N):
                try:
                    p_logpdf_x_new[i] = self.stan_model.log_density(x[i])
                except:
                    p_logpdf_x_new[i] = -np.inf
            return np.array(p_logpdf_x_new)
    
    def logpdfgrad(self, x, phi=1.0, adjust_transform=True):
        """
        Calculate the gradient of the log density of the target distribution at x

        Args:
            x: Unconstrained parameter values to evaluate the gradient of the log density at x
            phi: The temperature of the target distribution, default is no tempering, i.e. phi = 1.0

        Returns:
            The gradient of the log density of the target distribution at x (or a 2D array of gradients if x is a 2D array of samples)
        """

        if self.data_path and phi != self.last_phi:
            self._update_phi(phi)

        if x.ndim == 1:
            grad_x = np.zeros(self.dim)
            try:
                self.stan_model.log_density_gradient(x, out=grad_x)
            except:
                grad_x = np.full(self.dim, -np.inf)
            return np.array(grad_x)
        else:
            N = x.shape[0]
            p_logpdf_x_new = np.zeros((N, self.dim))
            for i in range(N):
                try:
                    self.stan_model.log_density_gradient(x[i], out=p_logpdf_x_new[i])
                except:
                    p_logpdf_x_new[i] = np.full(self.dim, -np.inf)
            return np.array(p_logpdf_x_new)
        

    def constrain(self, x, include_tparams=True, include_gqs=True):
        """
        Constrain the parameters to the support of the target distribution

        Args:
            x: Unconstrained parameter values to constrain
            include_tparams: Whether to include the transformed parameters
            include_gqs: Whether to include the generated quantities

        Returns:
            The constrained parameter values (or a 2D array of constrained parameters if x is a 2D array of samples)
        """

        stan_rng = self.stan_model.new_rng(seed=0)
        if x.ndim == 1:
            try:
                cpar = self.stan_model.param_constrain(x, include_tp=include_tparams, include_gq=include_gqs, rng=stan_rng)
                return np.array(cpar)
            except:
                return np.full(self.constrained_dim, 0.0)
        else:
            constrained_samples = np.zeros([x.shape[0], self.constrained_dim])
            for i in range(x.shape[0]):
                try:
                    constrained_samples[i] = self.stan_model.param_constrain(x[i], include_tp=include_tparams, include_gq=include_gqs, rng=stan_rng)
                except:
                    constrained_samples[i] = np.full(self.constrained_dim, 0.0)
            return np.array(constrained_samples)

    def _update_phi(self, phi):
        """
        Update the temperature of the target distribution in the json file
        and reload the model

        Args:
            phi: The new temperature of the target distribution

        """
        self.last_phi = phi

        # Open json file and change phi
        with open(self.data_path, "r") as f:
            data = json.load(f)
        if "phi" in data:
            data["phi"] = phi
            with open(self.data_path, "w") as f:
                json.dump(data, f)
                f.flush()
                os.fsync(f.fileno())

        self.last_phi = phi

        # Reload model
        self.stan_model = bs.StanModel.from_stan_file(self.model_path, self.data_path if self.data_path else None)
