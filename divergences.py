#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
kl divergence calculation for normal and binomioal distribution

@author: javad
"""

import torch
import torch.nn.functional as F


def gaussian_kl(post_mean, post_std, prior_mean, prior_std):
    # KL between two Gaussians (for ONE feature at a time)
    v = prior_std
    s = post_std
    m = post_mean
    mu = prior_mean
    term1 = torch.log(v / s)
    term2 = (s**2 + (m - mu)**2)
    term3 = (2 * v**2)
    return term1 + term2 / term3 - 0.5


def bernoulli_kl(p1, p2):
    # KL between two Bernoulli distributions
    # p1 = probability of the first  Bernoulli
    # p2 = probability of the second Bernoulli

    # make sure both are tensors so the checks and math work for floats too
    p1 = torch.as_tensor(p1)
    p2 = torch.as_tensor(p2)

    # input checks: both must be valid probabilities in [0, 1]
    if (p1 < 0).any() or (p1 > 1).any():
        raise ValueError("p1 must be in [0, 1]")
    if (p2 < 0).any() or (p2 > 1).any():
        raise ValueError("p2 must be in [0, 1]")

    return p1 * torch.log(p1 / p2) + (1 - p1) * torch.log((1 - p1) / (1 - p2))