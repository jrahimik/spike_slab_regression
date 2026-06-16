# Fully Differentiable Spike-and-Slab Variational Linear Regression

*Javad Rahimikollu*

A spike-and-slab Bayesian feature-selection model whose objective is **exactly
differentiable** — the discrete inclusion indicators are marginalized
analytically (no sampling, no Concrete/Gumbel relaxation), so the entire ELBO is
a smooth function of the variational parameters and trains with ordinary
gradient descent.

---

## Contents

- [1. Notation](#1-notation)
- [2. The Model](#2-the-model)
- [3. Inference](#3-inference)
- [4. Simulation Data-Generating Mechanism](#4-simulation-data-generating-mechanism)
- [Appendix A. Function Definitions](#appendix-a-function-definitions)
- [Appendix B. Why Mean-Field Independence Fails](#appendix-b-why-mean-field-independence-fails)
- [Appendix C. Derivation of the ELBO](#appendix-c-derivation-of-the-elbo)
- [Appendix D. Expected Log-Likelihood](#appendix-d-expected-log-likelihood)
- [Appendix E. Marginal Moments](#appendix-e-marginal-moments)
- [Appendix F. KL Decomposition](#appendix-f-kl-decomposition)
- [Appendix G. Unconstrained Optimization](#appendix-g-unconstrained-optimization)

---

## 1. Notation

Indices: $n = 1,\dots,N$ runs over observations, $k = 1,\dots,K$ over features.

**Data and dimensions**

| Symbol | Meaning |
|---|---|
| $N,\ K$ | number of observations; number of candidate features |
| $\mathbf{x}_n \in \mathbb{R}^K$ | covariate vector for observation $n$ |
| $\mathbf{X} \in \mathbb{R}^{N\times K}$ | design matrix stacking all $\mathbf{x}_n$ |
| $y_n \in \mathbb{R}$ | response for observation $n$ |
| $\sigma_{\text{obs}}^2$ | observation-noise variance |

**Latent variables (inferred)**

| Symbol | Meaning |
|---|---|
| $\gamma_k \in \{0,1\}$ | inclusion indicator: 1 if feature $k$ is active |
| $\beta_k \in \mathbb{R}$ | regression coefficient for feature $k$ |
| $\boldsymbol{\beta} \in \mathbb{R}^K$ | full coefficient vector |

**Fixed hyperparameters**

| Symbol | Meaning |
|---|---|
| $\alpha \in (0,1)$ | prior inclusion probability (sparsity target), *fixed* |
| $\sigma_\beta$ | slab scale: prior std of an active coefficient, *fixed* |
| $\mu_k$ | slab mean (often 0) |
| $\varepsilon$ | spike width: tiny std pinning inactive coefficients to 0 |

**Variational parameters (optimized)**

| Symbol | Meaning |
|---|---|
| $\text{pip}_k \in (0,1)$ | posterior inclusion probability (PIP), $\text{pip}_k = \sigma(\lambda_k)$ |
| $\lambda_k \in \mathbb{R}$ | unconstrained logit for $\text{pip}_k$ |
| $m_k \in \mathbb{R}$ | posterior mean of $\beta_k$ |
| $s_k > 0$ | posterior std of $\beta_k$, $s_k = \text{softplus}(\rho_k)$ |
| $\rho_k \in \mathbb{R}$ | unconstrained parameter for $s_k$ |

All function definitions ($\sigma$, softplus, the KL formulas) are in
[Appendix A](#appendix-a-function-definitions); all derivations are in
Appendices C–G.

---

## 2. The Model

The generative model places a spike-and-slab prior on each coefficient, gated by
a Bernoulli inclusion indicator, with a Gaussian likelihood on the response. The
**true (generative) distributions** are:

$$\gamma_k \mid \alpha \sim \text{Bernoulli}(\alpha)$$

$$\beta_k \mid \gamma_k \sim \gamma_k\,\mathcal{N}(\mu_k, \sigma_\beta^2) + (1-\gamma_k)\,\mathcal{N}(0, \varepsilon^2)$$

$$y_n \mid \mathbf{x}_n, \boldsymbol{\beta} \sim \mathcal{N}(\mathbf{x}_n^\top \boldsymbol{\beta},\ \sigma_{\text{obs}}^2)$$

An active feature ($\gamma_k = 1$) draws its coefficient from the broad **slab**
$\mathcal{N}(\mu_k, \sigma_\beta^2)$; an inactive one ($\gamma_k = 0$) from the
razor-thin **spike** $\mathcal{N}(0, \varepsilon^2)$. Here $\alpha$ and
$\sigma_\beta$ are **fixed** hyperparameters, not inferred.

**Joint distribution.** Reading the dependency structure
($\alpha \to \gamma_k$, $(\sigma_\beta, \gamma_k) \to \beta_k$,
$(\mathbf{x}_n, \boldsymbol{\beta}) \to y_n$), the joint factorizes as one
likelihood factor per observation and one prior factor per feature:

$$p(y, \boldsymbol{\beta}, \gamma \mid \mathbf{X}) = \underbrace{\prod_{n=1}^N p(y_n \mid \mathbf{x}_n, \boldsymbol{\beta})}_{\text{likelihood}} \underbrace{\prod_{k=1}^K p(\beta_k \mid \gamma_k)\, p(\gamma_k \mid \alpha)}_{\text{prior}}$$

Each likelihood factor conditions on the **entire** vector $\boldsymbol{\beta}$
through $\mathbf{x}_n^\top \boldsymbol{\beta} = \sum_k x_{nk}\beta_k$, not a
single $\beta_k$.

---

## 3. Inference

### 3.1 The variational distribution

We approximate the intractable posterior
$p(\boldsymbol{\beta}, \gamma \mid y, \mathbf{X})$ with a variational
distribution $q$. Within a feature we keep the $\gamma_k \to \beta_k$ dependence
(see [Appendix B](#appendix-b-why-mean-field-independence-fails)); across
features we assume independence. The variational distributions are:

$$q(\gamma_k) = \text{Bernoulli}(\text{pip}_k), \qquad \text{pip}_k = \sigma(\lambda_k)$$

$$q(\beta_k \mid \gamma_k = 1) = \mathcal{N}(m_k, s_k^2), \qquad s_k = \text{softplus}(\rho_k)$$

$$q(\beta_k \mid \gamma_k = 0) = \mathcal{N}(0, \varepsilon^2) \quad \text{(the prior spike)}$$

The free parameters optimized are $\{\lambda_k, m_k, \rho_k\}_{k=1}^K$; the
constrained quantities $\text{pip}_k \in (0,1)$ and $s_k > 0$ come from the link
functions $\sigma$ and softplus ([Appendix A](#appendix-a-function-definitions)).
The quantity $\text{pip}_k$ is the **posterior inclusion probability (PIP)**: the
model's belief, after seeing the data, that feature $k$ is active.

### 3.2 End-to-end differentiability via analytic marginalization

A central design choice: the discrete indicator $\gamma_k$ is **not sampled**.
Sampling a $\text{Bernoulli}(\text{pip}_k)$ would yield a hard 0/1 whose
derivative does not exist, breaking gradient flow.

Instead we **average over** the two cases. The indicator $\gamma_k$ can only be 1
(feature on, probability $\text{pip}_k$) or 0 (feature off, probability
$1 - \text{pip}_k$). For any quantity we compute it in each case and weight by the
probabilities. Taking the coefficient as the example:

$$\mathbb{E}[\beta_k] = \underbrace{m_k}_{\text{if on}} \times \underbrace{\text{pip}_k}_{\text{prob on}} + \underbrace{0}_{\text{if off}} \times \underbrace{(1-\text{pip}_k)}_{\text{prob off}} = \text{pip}_k\, m_k$$

> **Why this is differentiable (and exact).** A Bernoulli *sample* is not
> differentiable; a Bernoulli *probability* is. The computation uses only the
> probability $\text{pip}_k = \sigma(\lambda_k)$, inside an analytic sum over the
> two states of $\gamma_k$ — so gradients flow exactly. A Concrete / Gumbel-Softmax
> relaxation $\sigma((\lambda_k + L)/\tau)$ with sampled Logistic noise $L$ would
> be needed *only* for a non-Gaussian likelihood; it is **not** used here, and the
> present approach is both exact and zero-variance.

### 3.3 Marginal moments of the coefficients

Marginalizing $\gamma_k$ (spike at zero) gives the gated moments the likelihood
needs (derivation in [Appendix E](#appendix-e-marginal-moments)):

$$\mathbb{E}_q[\beta_k] = \text{pip}_k\, m_k, \qquad \text{Var}_q[\beta_k] = \text{pip}_k(s_k^2 + m_k^2) - (\text{pip}_k\, m_k)^2$$

The factor $\text{pip}_k$ **gates** each feature: its coefficient enters the model
only to the extent the feature is included.

### 3.4 The objective (ELBO)

Inference maximizes the evidence lower bound — the expected log-likelihood minus
the KL to the prior (derived in [Appendix C](#appendix-c-derivation-of-the-elbo)):

$$\mathcal{L} = \mathbb{E}_{q(\boldsymbol{\beta})}\big[\log p(y \mid \mathbf{X}, \boldsymbol{\beta})\big] - \text{KL}\big(q(\boldsymbol{\beta}, \gamma)\,\|\,p(\boldsymbol{\beta}, \gamma)\big)$$

Because $\alpha$ and $\sigma_\beta$ are fixed, they contribute no KL term. The
explicit objective is:

$$\mathcal{L} = \sum_n \left[ -\tfrac{1}{2}\log(2\pi\sigma_{\text{obs}}^2) - \frac{(y_n - \mu_n)^2 + v_n}{2\sigma_{\text{obs}}^2} \right] - \sum_k \Big[ \text{KL}_{\text{Bern}}(\text{pip}_k \,\|\, \alpha) + \text{pip}_k\, \text{KL}_{\text{slab},k} \Big]$$

with predictive moments $\mu_n = \sum_k x_{nk}\, \text{pip}_k\, m_k$ and
$v_n = \sum_k x_{nk}^2\, \text{Var}_q[\beta_k]$. Training minimizes
$\mathcal{J} = -\mathcal{L}$ by gradient descent on $\{\lambda_k, m_k, \rho_k\}$.

### 3.5 Read-outs

After training, feature $k$ is **selected** if its PIP exceeds a threshold, and
its effective coefficient is the gated mean:

$$\text{selected}_k = [\,\text{pip}_k > \tfrac{1}{2}\,], \qquad \hat{\beta}_k = \text{pip}_k\, m_k$$

---

## 4. Simulation Data-Generating Mechanism

To evaluate the method we generate synthetic data with known ground truth.

**Quantities held fixed:** $N$ (observations), $K$ (candidate features),
$K_{\text{act}}$ (active features), $\sigma_{\text{noise}}$ (noise std), the
effect-size pool $\{c_j\}$, and the RNG seed.

**Generative procedure:**

1. **Design matrix.** $X_{nk} \sim \mathcal{N}(0,1)$ i.i.d.
2. **True support.** Choose $\mathcal{A} \subset \{1,\dots,K\}$ with
   $|\mathcal{A}| = K_{\text{act}}$ uniformly at random; set
   $\gamma_k^{\text{true}} = \mathbb{1}[k \in \mathcal{A}]$.
3. **True coefficients.** Active features get effect sizes from $\{c_j\}$
   (varied sign/magnitude); inactive features get exactly zero, so
   $\boldsymbol{\beta}_{\text{true}}$ is exactly sparse.
4. **Responses.** $y_n = \mathbf{x}_n^\top \boldsymbol{\beta}_{\text{true}} + \epsilon_n$,
   with $\epsilon_n \sim \mathcal{N}(0, \sigma_{\text{noise}}^2)$.

**Difficulty controls:** the noise level sets the signal-to-noise ratio;
sparsity ($K_{\text{act}}$) makes the support denser or sparser; effect size
(scaling $\{c_j\}$) moves the signal toward or away from the noise floor; and
dimensionality ($N$ relative to $K$) probes the small-sample regime.

---

## Appendix A. Function Definitions

**Sigmoid (logistic) link.** Maps $\mathbb{R} \to (0,1)$:

$$\sigma(x) = \frac{1}{1 + e^{-x}}, \qquad \sigma'(x) = \sigma(x)(1 - \sigma(x))$$

**Softplus link.** Maps $\mathbb{R} \to (0, \infty)$:

$$\text{softplus}(x) = \log(1 + e^x), \qquad \text{softplus}'(x) = \sigma(x)$$

Computed stably as $\text{softplus}(x) = \max(x, 0) + \log(1 + e^{-|x|})$, which
avoids overflow; its derivative is the sigmoid (the kink in $\max$ cancels, so the
function is smooth everywhere).

**Stable log-sigmoid.** For log-probabilities:

$$\log \sigma(x) = -\text{softplus}(-x), \qquad \log(1 - \sigma(x)) = -\text{softplus}(x)$$

These never form the underflowing probability $\sigma(x)$, so they avoid
$\log 0 = -\infty$ and the resulting NaN gradient.

**Gaussian–Gaussian KL.** For $q = \mathcal{N}(m, s^2)$, $p = \mathcal{N}(\mu, \nu^2)$:

$$\text{KL}\big(\mathcal{N}(m, s^2)\,\|\,\mathcal{N}(\mu, \nu^2)\big) = \log\frac{\nu}{s} + \frac{s^2 + (m-\mu)^2}{2\nu^2} - \frac{1}{2}$$

**Bernoulli–Bernoulli KL.** For $q = \text{Bernoulli}(\text{pip})$, $p = \text{Bernoulli}(\alpha)$:

$$\text{KL}_{\text{Bern}}(\text{pip} \,\|\, \alpha) = \text{pip}\log\frac{\text{pip}}{\alpha} + (1-\text{pip})\log\frac{1-\text{pip}}{1-\alpha}$$

**Slab KL** (shorthand in §3.4):

$$\text{KL}_{\text{slab},k} = \log\frac{\sigma_\beta}{s_k} + \frac{s_k^2 + (m_k - \mu_k)^2}{2\sigma_\beta^2} - \frac{1}{2}$$

---

## Appendix B. Why Mean-Field Independence Fails

A naive posterior would assume full independence,
$q(\beta_k, \gamma_k) = q(\beta_k)q(\gamma_k)$, with a single Gaussian
$q(\beta_k) = \mathcal{N}(m_k, s_k^2)$ regardless of $\gamma_k$. This fails for
two reasons.

**A single Gaussian cannot represent a spike-and-slab posterior.** The true
posterior over $\beta_k$ is **bimodal**: mass near zero (feature off) and mass
around the active value (feature on). A unimodal Gaussian must either straddle
both modes (placing its mean in the low-density valley between them) or collapse
onto one and discard the other — either way misrepresenting the include/exclude
structure the model exists to express.

**Independence destroys the $\gamma_k \to \beta_k$ link.** The generative model
makes $\beta_k$ depend on $\gamma_k$ (slab if on, spike if off). Writing
$q(\beta_k)q(\gamma_k)$ asserts the coefficient carries no information about
inclusion, which is false: $\beta_k \approx 0$ is evidence the feature is off.
The fix is the conditional form $q(\gamma_k)q(\beta_k \mid \gamma_k)$, whose
marginal
$q(\beta_k) = \text{pip}_k\,\mathcal{N}(m_k, s_k^2) + (1-\text{pip}_k)\,\mathcal{N}(0, \varepsilon^2)$
is a genuine bimodal mixture. Independence is kept only **across** features.

---

## Appendix C. Derivation of the ELBO

For any $q$, multiply and divide by $q$ and apply Jensen's inequality:

$$\log p(y \mid \mathbf{X}) = \log \int \sum_\gamma p(y, \boldsymbol{\beta}, \gamma \mid \mathbf{X})\, d\boldsymbol{\beta}$$

$$= \log \int \sum_\gamma \frac{p(y, \boldsymbol{\beta}, \gamma \mid \mathbf{X})}{q(\boldsymbol{\beta}, \gamma)}\, q(\boldsymbol{\beta}, \gamma)\, d\boldsymbol{\beta} \qquad \text{(multiply by } q/q = 1\text{)}$$

$$= \log \mathbb{E}_q\!\left[\frac{p(y, \boldsymbol{\beta}, \gamma \mid \mathbf{X})}{q(\boldsymbol{\beta}, \gamma)}\right] \ge \mathbb{E}_q\!\left[\log \frac{p(y, \boldsymbol{\beta}, \gamma \mid \mathbf{X})}{q(\boldsymbol{\beta}, \gamma)}\right] =: \mathcal{L} \qquad \text{(Jensen)}$$

The gap is the **posterior** KL. Substituting Bayes' rule
$p(\boldsymbol{\beta}, \gamma \mid y, \mathbf{X}) = p(y \mid \mathbf{X}, \boldsymbol{\beta})\, p(\boldsymbol{\beta}, \gamma) / p(y \mid \mathbf{X})$:

$$\text{KL}\big(q \,\|\, p(\boldsymbol{\beta}, \gamma \mid y, \mathbf{X})\big) = \text{KL}\big(q \,\|\, p(\boldsymbol{\beta}, \gamma)\big) - \mathbb{E}_q[\log p(y \mid \mathbf{X}, \boldsymbol{\beta})] + \log p(y \mid \mathbf{X})$$

using $\mathbb{E}_q[\log p(y \mid \mathbf{X})] = \log p(y \mid \mathbf{X})$
(constant in the latents). Rearranging gives the evidence decomposition:

$$\log p(y \mid \mathbf{X}) = \mathcal{L} + \text{KL}\big(q(\boldsymbol{\beta}, \gamma) \,\|\, p(\boldsymbol{\beta}, \gamma \mid y, \mathbf{X})\big)$$

So maximizing $\mathcal{L}$ minimizes the KL to the true posterior. The ELBO
itself contains only the **prior** KL (computable); the **posterior** KL appears
only as this gap.

---

## Appendix D. Expected Log-Likelihood

Since the Gaussian log-density is quadratic in $\boldsymbol{\beta}$, its
expectation needs only the first two moments. With
$\eta_n = \mathbf{x}_n^\top \boldsymbol{\beta}$ and
$\mathbb{E}[(y - \eta)^2] = (y - \mathbb{E}\eta)^2 + \text{Var}(\eta)$:

$$\mathbb{E}_{q(\boldsymbol{\beta})}\big[\log p(y_n \mid \mathbf{x}_n, \boldsymbol{\beta})\big] = -\tfrac{1}{2}\log(2\pi\sigma_{\text{obs}}^2) - \frac{(y_n - \mu_n)^2 + v_n}{2\sigma_{\text{obs}}^2}$$

with $\mu_n = \sum_k x_{nk}\,\mathbb{E}_q[\beta_k]$ and (mean-field, no
cross-covariances) $v_n = \sum_k x_{nk}^2\,\text{Var}_q[\beta_k]$. No sampling is
required.

---

## Appendix E. Marginal Moments

By the law of total expectation over $q(\gamma_k)$ (spike at zero):

$$\mathbb{E}_q[\beta_k] = \text{pip}_k\, m_k + (1-\text{pip}_k)\cdot 0 = \text{pip}_k\, m_k$$

$$\mathbb{E}_q[\beta_k^2] = \text{pip}_k(m_k^2 + s_k^2) + (1-\text{pip}_k)\cdot 0 = \text{pip}_k(m_k^2 + s_k^2)$$

$$\text{Var}_q[\beta_k] = \mathbb{E}_q[\beta_k^2] - (\mathbb{E}_q[\beta_k])^2 = \text{pip}_k(s_k^2 + m_k^2) - (\text{pip}_k\, m_k)^2$$

The variance equals $\text{pip}_k\, s_k^2 + \text{pip}_k(1-\text{pip}_k)m_k^2$: the
within-slab variance plus the inclusion uncertainty propagated into $\beta_k$.

---

## Appendix F. KL Decomposition

Because the prior $p(\beta_k \mid \gamma_k)$ depends on $\gamma_k$, the joint KL
decomposes by the chain rule:

$$\text{KL}\big(q(\beta_k, \gamma_k) \,\|\, p(\beta_k, \gamma_k)\big) = \underbrace{\text{KL}\big(q(\gamma_k) \,\|\, p(\gamma_k)\big)}_{\text{KL}_{\text{Bern}}(\text{pip}_k \| \alpha)} + \underbrace{\mathbb{E}_{q(\gamma_k)}\big[\text{KL}(q(\beta_k \mid \gamma_k) \,\|\, p(\beta_k \mid \gamma_k))\big]}_{\text{pip}_k\, \text{KL}_{\text{slab},k}}$$

The indicator term is the Bernoulli KL. The coefficient term is the
$\text{pip}_k$-weighted Gaussian KL; the spike branch has
$\text{KL}_{\text{spike}} = \text{KL}(\mathcal{N}(0,\varepsilon^2)\,\|\,\mathcal{N}(0,\varepsilon^2)) = 0$,
so only $\text{pip}_k\, \text{KL}_{\text{slab},k}$ remains.

**Fixed hyperparameters contribute no term.** $\alpha$ and $\sigma_\beta$ have no
prior and no posterior — they are fixed, not inferred — so neither contributes a
KL term. They enter only as the constant sparsity target inside
$\text{KL}_{\text{Bern}}$ and the constant slab scale inside
$\text{KL}_{\text{slab},k}$.

---

## Appendix G. Unconstrained Optimization

The constrained quantities are reparameterized through the links of
[Appendix A](#appendix-a-function-definitions) so the optimizer works on free
real parameters:

$$\text{pip}_k = \sigma(\lambda_k) \in (0,1), \qquad s_k = \text{softplus}(\rho_k) \in (0,\infty), \qquad m_k \in \mathbb{R} \text{ (no link)}$$

Log-probabilities in $\text{KL}_{\text{Bern}}$ use the stable
$\log\sigma(\cdot) = -\text{softplus}(-\cdot)$ to avoid $\log 0$. After these
substitutions every term of $\mathcal{J} = -\mathcal{L}$ is a composition of
smooth elementary functions of $\{\lambda_k, m_k, \rho_k\}$ with no discrete node
and no sampling, so `loss.backward()` differentiates the objective directly. The
fixed $\alpha, \sigma_\beta$ need no links — they are constants.
