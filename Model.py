"""
spike and slab model

@author: javad
"""

import torch
import torch.nn as nn
from divergences import gaussian_kl, bernoulli_kl


class SpikeSlabVI(nn.Module):

    def __init__(self, X, alpha=0.25, slab_std=3.0, obs_std=0.3):
        super().__init__()

        # number of features = number of columns in the data
        K = X.shape[1]
        self.K = K

        # save the fixed settings
        self.alpha = alpha
        self.slab_std = slab_std
        self.obs_var = obs_std**2

        # the numbers we learn (one per feature), all starting at zero
        self.inclusion_logit = nn.Parameter(torch.zeros(K))
        self.coef_mean = nn.Parameter(torch.zeros(K))
        self.coef_rho = nn.Parameter(torch.zeros(K))

    def get_inclusion_prob(self):
        # logit -> probability between 0 and 1
        return torch.sigmoid(self.inclusion_logit)

    def get_coef_std(self):
        # rho -> positive standard deviation
        return torch.nn.functional.softplus(self.coef_rho)

    def get_effective_mean(self):
        # E[beta] = varpi * m, computed one feature at a time
        varpi = self.get_inclusion_prob()
        m = self.coef_mean
        result = []
        for k in range(self.K):
            result.append(varpi[k] * m[k])
        return torch.stack(result)

    def get_effective_var(self):
        # Var[beta] = varpi*(s^2 + m^2) - (varpi*m)^2, one feature at a time
        varpi = self.get_inclusion_prob()
        m = self.coef_mean
        s = self.get_coef_std()
        result = []
        for k in range(self.K):
            var_k = varpi[k] * (s[k]**2 + m[k]**2) - (varpi[k] * m[k])**2
            result.append(var_k)
        return torch.stack(result)

    def get_expected_log_likelihood(self, X, y):
        # how well the model fits the data
        beta_mean = self.get_effective_mean()
        beta_var = self.get_effective_var()

        # predictions for all data points at once
        pred_mean = X @ beta_mean        # X times effective mean
        pred_var = (X**2) @ beta_var     # X^2 times effective variance

        N = X.shape[0]
        total = 0.0
        # add up the log-likelihood one data point at a time
        for n in range(N):
            const = -0.5 * torch.log(2 * torch.tensor(torch.pi) * self.obs_var)
            fit = -((y[n] - pred_mean[n])**2 + pred_var[n]) / (2 * self.obs_var)
            total = total + const + fit
        return total

    def get_kl(self):
        # joint (beta, gamma) KL, summed over features one at a time
        varpi = self.get_inclusion_prob()
        s = self.get_coef_std()
        m = self.coef_mean

        total_kl = 0.0
        for k in range(self.K):
            # gamma part for feature k (compare two probabilities)
            kl_gamma_k = bernoulli_kl(varpi[k], self.alpha)

            # beta part for feature k (weighted by inclusion prob)
            prior_mean = torch.tensor(0.0)
            prior_std = torch.tensor(self.slab_std)
            kl_beta_k = varpi[k] * gaussian_kl(m[k], s[k], prior_mean, prior_std)

            total_kl = total_kl + kl_gamma_k + kl_beta_k
        return total_kl

    def get_loss(self, X, y):
        # loss = KL - expected log-likelihood (the negative ELBO)
        return self.get_kl() - self.get_expected_log_likelihood(X, y)

    def get_selected_features(self, threshold=0.5):
        # which features are "on"
        return self.get_inclusion_prob() > threshold

    def get_point_estimate(self):
        # final coefficient values for prediction: varpi * m
        return self.get_effective_mean()