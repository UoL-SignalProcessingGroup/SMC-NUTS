import os
import json
import bridgestan as bs
import autograd.numpy as np


class StanModel():
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

    def logpdf(self, upar, phi=1.0, adjust_transform=True):
        if self.data_path and phi != self.last_phi:
            self._update_phi(phi)

        # If upar is a 1D array, calculate the logpdf of upar
        if upar.ndim == 1:
            try:
                return self.stan_model.log_density(upar)
            except:
                return -np.inf
        else:
            N = upar.shape[0]
            p_logpdf_x_new = np.zeros(N)
            for i in range(N):
                try:
                    p_logpdf_x_new[i] = self.stan_model.log_density(upar[i])
                except:
                    p_logpdf_x_new[i] = -np.inf
            return np.array(p_logpdf_x_new)
    
    def logpdfgrad(self, upar, phi=1.0, adjust_transform=True):
        if self.data_path and phi != self.last_phi:
            self._update_phi(phi)

        if upar.ndim == 1:
            grad_x = np.zeros(self.dim)
            try:
                self.stan_model.log_density_gradient(upar, out=grad_x)
            except:
                grad_x = np.full(self.dim, -np.inf)
            return np.array(grad_x)
        else:
            N = upar.shape[0]
            p_logpdf_x_new = np.zeros((N, self.dim))
            for i in range(N):
                try:
                    self.stan_model.log_density_gradient(upar[i], out=p_logpdf_x_new[i])
                except:
                    p_logpdf_x_new[i] = np.full(self.dim, -np.inf)
            return np.array(p_logpdf_x_new)

    def constrain(self, upar, include_tparams=True, include_gqs=True):
        stan_rng = self.stan_model.new_rng(seed=0)
        if upar.ndim == 1:
            try:
                cpar = self.stan_model.param_constrain(upar, include_tp=include_tparams, include_gq=include_gqs, rng=stan_rng)
                return np.array(cpar)
            except:
                return np.full(self.constrained_dim, 0.0)
        else:
            constrained_samples = np.zeros([upar.shape[0], self.constrained_dim])
            for i in range(upar.shape[0]):
                try:
                    constrained_samples[i] = self.stan_model.param_constrain(upar[i], include_tp=include_tparams, include_gq=include_gqs, rng=stan_rng)
                except:
                    constrained_samples[i] = np.full(self.constrained_dim, 0.0)
            return np.array(constrained_samples)

    def _update_phi(self, phi):
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
