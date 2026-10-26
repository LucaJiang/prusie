# Statistical model

Fine-mapping aims to identify variants that affect a trait, but linkage
disequilibrium (LD) can make several nearby variants consistent with the same
association signal. prusie uses SuSiE (Sum of Single Effects) to model multiple
effects while quantifying uncertainty about which variants carry them.

SuSiE is a Bayesian variable selection model for linear regression with Gaussian
errors. It represents the vector of variant effects as a sum of single-effect
components, each with at most one nonzero effect.

prusie can fit this model using individual-level data, sufficient statistics
computed from centered data, or association summary statistics together with an
LD correlation matrix. The sections below develop the model with Gaussian effect
priors, starting with the data and notation; [additional model options](#additional-model-options)
are introduced later.

## Data, notation and scale

Let $n$ be the study sample size, $p$ the number of variants and $L$ the maximum
number of single-effect components. The centered design matrix
$\mathbf X\in\mathbb R^{n\times p}$ contains the predictors and the centered
response is $\mathbf y\in\mathbb R^n$. Their sufficient statistics are

$$
\mathbf Q=\mathbf X^T\mathbf X,\qquad
\mathbf g=\mathbf X^T\mathbf y,\qquad s_y=\mathbf y^T\mathbf y.
$$

The LD correlation matrix $\mathbf R$ refers to the same variant order and
effect alleles. The residual variance is $\sigma^2$; single effect $\ell$ has
prior effect variance $V_\ell$ (written $\sigma_0^2$ in some references), and
$\pi_j$ is the prior probability that it is located at variant $j$. prusie uses
the same prior weights for all single effects.

With `standardize=True` (the default), each centered predictor column is
divided by its sample standard deviation
$d_j=\sqrt{Q_{jj}/(n-1)}$. Each nonconstant column then has unit sample
variance, while the response remains in its supplied units. For a constant
column, the divisor is set to one to avoid division by zero. With
`standardize=False`, all divisors are one and the predictor units are unchanged.

The corresponding sufficient statistics used for fitting are
$\mathbf D^{-1}\mathbf Q\mathbf D^{-1}$ and $\mathbf D^{-1}\mathbf g$, where
$\mathbf D=\operatorname{diag}(d_1,\ldots,d_p)$. In the regression and updates below,
the predictor matrix is on this **working scale**, and
$\mathbf Q$ and $\mathbf g$ denote the working sufficient statistics.

Effect estimates are computed on the working predictor scale. The `coef`
property divides each posterior mean coefficient by its stored predictor scale
factor to express the effect per unit of the input predictor. For summary-statistic
inputs, these units depend on how the regression quantities are constructed;
see [From summaries to working sufficient statistics](#from-summaries-to-working-sufficient-statistics).

Standardization is separate from centering and covariate adjustment. Supplied
summary statistics and sufficient statistics must already reflect consistent
centering and any covariate adjustment; the fitting routines do not estimate
additional covariate effects.

## A sum of single effects

The sampling model is Gaussian regression:

$$
\begin{aligned}
\mathbf y&=\mathbf X\mathbf b+\boldsymbol\varepsilon,
&\boldsymbol\varepsilon&\sim N(\mathbf0,\sigma^2\mathbf I_n),\\
\mathbf b&=\sum_{\ell=1}^{L}\mathbf b_\ell.
\end{aligned}
$$

Each single effect, indexed by $\ell=1,\ldots,L$, is located at one variant
and has a Gaussian effect size:

$$
\begin{aligned}
\gamma_\ell&\sim\operatorname{Categorical}(\pi_1,\ldots,\pi_p),\\
a_\ell&\sim N(0,V_\ell),\\
\mathbf b_\ell&=a_\ell\mathbf e_{\gamma_\ell}.
\end{aligned}
$$

For component $\ell$, $\gamma_\ell$ selects a variant from $1,\ldots,p$, and
$a_\ell$ is its effect size. The vector $\mathbf e_j$ has a one in position
$j$ and zeros elsewhere, so $\mathbf b_\ell$ places the effect at the selected
variant and assigns zero to every other variant.

The prior probability of selecting variant $j$ is $\pi_j$. By default, all
variants are equally likely ($\pi_j=1/p$); `prior_weights` allows unequal
probabilities. The effect-size prior is centered at zero with variance
$V_\ell$. In this Gaussian model, its initial value is `scaled_prior_variance`
times the response sample variance $s_y/(n-1)$. This parameter defaults to
0.2 and accepts either a shared value or a separate value for each component.

$L$ sets the maximum number of single-effect components available to the model.
A component with $V_\ell=0$ contributes no effect, and several components can
select the same variant. Choosing $L$ therefore does not assume that exactly
$L$ distinct variants affect the trait. Credible sets summarize uncertainty
about component locations; sets from inactive components or with low LD purity
are excluded by default, and duplicate sets are removed. The number of reported
credible sets therefore need not equal $L$ or the number of causal variants.

## Single-effect posteriors and effect means

SuSiE uses a variational approximation that factorizes over the single effects:

$$
q(\mathbf b_1,\ldots,\mathbf b_L)=\prod_{\ell=1}^{L}q_\ell(\mathbf b_\ell).
$$

Within a single effect, the uncertainty about its location and about its size
remain coupled. The returned arrays represent

$$
\begin{aligned}
\alpha_{\ell j}&=q_\ell(\gamma_\ell=j),\\
\mu_{\ell j}&=\mathbb E_q[a_\ell\mid\gamma_\ell=j],\\
M^{(2)}_{\ell j}&=\mathbb E_q[a_\ell^2\mid\gamma_\ell=j]
=\mu_{\ell j}^2+\tau_{\ell j}^2.
\end{aligned}
$$

The result arrays `alpha`, `mu` and `mu2` store the inclusion probabilities,
conditional posterior means and conditional posterior second moments defined
above. Rows correspond to single-effect components and columns to variants;
$\tau_{\ell j}^2$ denotes the conditional posterior variance.

The posterior mean contribution of component $\ell$ is obtained by weighting
its conditional means by the inclusion probabilities:
$\mathbf m_\ell=\boldsymbol\alpha_\ell\odot\boldsymbol\mu_\ell$, where
$\odot$ denotes elementwise multiplication. Summing over components gives the
total posterior mean $\mathbf m=\sum_\ell\mathbf m_\ell$. The `coef` property
returns $m_j/d_j$ on the input predictor scale. For RSS fits, the units depend
on the input route, as described under
[From summaries to working sufficient statistics](#from-summaries-to-working-sufficient-statistics).

## One single-effect regression

To update effect $\ell$, the posterior means of the other effects are held
fixed and

$$
\begin{aligned}
\mathbf r_\ell&=\mathbf g-\mathbf Q\sum_{k\ne\ell}\mathbf m_k,\\
\widehat b_{\ell j}&=r_{\ell j}/Q_{jj},\\
s_j^2&=\sigma^2/Q_{jj}.
\end{aligned}
$$

The **crossproduct residual** $\mathbf r_\ell$ contains one entry per variant.
Each entry $r_{\ell j}$ is the inner product of predictor column $j$ with the
response residual after subtracting the posterior mean contributions of all
components except $\ell$. For $Q_{jj}>0$, the Gaussian single-effect
regression (SER) gives, for each variant, the Bayes factor of a nonzero effect
with prior $N(0,V_\ell)$ against no effect:

$$
\log\mathrm{BF}_{\ell j}
=-\frac12\log\left(1+\frac{V_\ell}{s_j^2}\right)
+\frac{\widehat b_{\ell j}^{\,2}V_\ell}
{2s_j^2(s_j^2+V_\ell)}.
$$

Combining the Bayes factor with the variant prior gives

$$
\begin{aligned}
\alpha_{\ell j}&=\frac{\pi_j\mathrm{BF}_{\ell j}}
{\sum_k\pi_k\mathrm{BF}_{\ell k}},\\
\mu_{\ell j}&=\frac{V_\ell}{V_\ell+s_j^2}\widehat b_{\ell j},\\
\tau_{\ell j}^2&=\frac{V_\ell s_j^2}{V_\ell+s_j^2}.
\end{aligned}
$$

The probabilities $\alpha_{\ell j}$ describe uncertainty about the effect's
location, combining the prior weights with evidence from the current residual.
Correlated variants with similar evidence can share the posterior probability.
Conditional on selecting variant $j$, the posterior mean $\mu_{\ell j}$ shrinks
its residual-based estimate $\widehat b_{\ell j}$ toward zero. Shrinkage is
stronger when the sampling variance $s_j^2$ is large relative to the prior
variance $V_\ell$.

`lbf_variable` stores $\log\mathrm{BF}_{\ell j}$ for each variant before prior
weighting. `lbf` stores $\log\sum_j\pi_j\mathrm{BF}_{\ell j}$ for each component,
summarizing its evidence after averaging over the possible variant locations.

These formulas assume $V_\ell>0$ and $Q_{jj}>0$.
[Inputs](inputs.md#priors-and-zero-information-columns) describes how prusie
smooths prior weights and handles columns without information.

## Sequential IBSS and variance estimation

Iterative Bayesian stepwise selection (IBSS) updates one single-effect component
at a time. A **sweep** is one pass through all components. Each update uses the
latest posterior means of the other components: those already updated in the
current sweep and those still carrying their values from the previous sweep.
At a cold start, the posterior means are zero, each row of `alpha` is uniform,
and the prior and residual variances take their initial values.

```text
initialize component means, variances and their matrix products
for each sweep, up to max_iter:
    for each component in order:
        remove its old contribution from the fitted crossproduct
        form its crossproduct residual
        select V and compute the SER posterior (EM updates V after the posterior)
        store the component evidence, KL and new matrix product
    if requested: check prior variance using the current residual variance
    evaluate expected residual sum of squares and ELBO
    record diagnostics and apply the stopping rule
    if converged, an eligible cycle stop, or numerical failure: stop
    if requested: update residual variance
trim components below prior_tol and construct summaries
```

### Prior variance estimation

When `estimate_prior_variance=True`, the prior variance is estimated within
each component update. `estimate_prior_method` selects the procedure:

- **`optim`** maximizes the SER log marginal likelihood
  $\log\sum_j\pi_j\mathrm{BF}_{\ell j}(V)$ over $\log V\in[-30,15]$ using
  Brent's method and the smoothed prior weights. The candidate replaces the
  current variance only if its log marginal likelihood is at least as large.
  The selected variance is then compared with zero: it is retained only if
  its log marginal likelihood exceeds that at $V=0$ by more than
  `check_null_threshold`; otherwise it is set to zero.
- **`simple`** compares the current variance with zero using the same
  `check_null_threshold` rule, without searching for a new positive variance.
- **`EM`** first computes the posterior using the current $V_\ell$, then sets
  $V_\ell\leftarrow\sum_j\alpha_{\ell j}M^{(2)}_{\ell j}$ for the next
  update of that component. The stored Bayes factors and KL divergence refer to the
  posterior computed before this variance update. EM does not perform the
  comparison with zero.

With `estimate_prior_variance=False`, the supplied initial prior variances are
kept during fitting. Final trimming of variances below `prior_tol` still applies.

### ELBO and residual variance

After each sweep, IBSS computes the expected residual sum of squares (ER2) and
the evidence lower bound (ELBO). ER2 averages the squared residual over the
posterior, accounting for uncertainty in the effects;
[Optimization](optimization.md#retained-component-products) describes its
calculation from `mu2` and the stored matrix products. The ELBO combines the
expected log likelihood with a penalty for divergence from the prior:

$$
\mathcal L(q)=\mathbb E_q[\log p(\mathbf y\mid\mathbf b,\sigma^2)]
-\sum_{\ell=1}^{L}\operatorname{KL}\{q_\ell(\mathbf b_\ell)\|p_\ell(\mathbf b_\ell)\}.
$$

The ELBO is evaluated at the current $\sigma^2$, and the stopping rule is
checked before updating residual variance. If the stopping check does not
terminate the fit and `estimate_residual_variance=True`, the ordinary Gaussian
model updates $\sigma^2\leftarrow\mathrm{ER2}/n$, constrained by
`residual_variance_lowerbound` and `residual_variance_upperbound`.

This update also runs after the last permitted sweep if the stopping check has
not terminated the fit. Consequently, a fit that reaches `max_iter` can return
a `sigma2` different from the value used for its last ELBO; a converged fit
skips the update. Final trimming of small prior variances leaves the recorded
ELBO and fitted crossproduct unchanged.
[Results](results.md#iterations-and-finalization) explains these returned quantities.

### Stopping rules

The first sweep has no preceding state, so every rule is first evaluated after
the second sweep (`min_iter`, default 2). The default ELBO rule
(`convergence_method="elbo"`) declares convergence when

$$
0\le\mathcal L^{(t)}-\mathcal L^{(t-1)}<\mathtt{tol}.
$$

A decrease does not satisfy it, and repeated posterior states are recorded only
as diagnostics.

The PIP rule follows susieR and requires both

$$
\max_{\ell,j}\bigl|\alpha^{(t)}_{\ell j}-\alpha^{(t-1)}_{\ell j}\bigr|<\mathtt{tol}
\quad\text{and}\quad
\max_j\bigl|\mathrm{PIP}^{(t)}_j-\mathrm{PIP}^{(t-1)}_j\bigr|<\mathtt{tol}.
$$

Each PIP uses the components whose $V_\ell$ exceeds `prior_tol` in the sweep in
which the PIP was computed; as in susieR, the state preceding the first check is
evaluated with the variances current at that check. If the condition fails but
alpha and PIP both match the state $k$ sweeps earlier within `tol`, for some
$2\le k\le5$, the fit is declared converged and alpha is replaced by the average
of the $k$ states in the cycle. The PIP rule is used:

- with `convergence_method="pip"`;
- for slot-activity priors and for NIG with more than one effect;
- by `susie_rss` with `R_finite` or `R_mismatch`, unless a `convergence`
  configuration is supplied, and always by multi-panel RSS;
- for a sweep whose ELBO is NaN under the default ELBO rule, when no
  `convergence` configuration is supplied.

An infinite ELBO raises an error. A `ConvergenceConfig` passed as `convergence`
selects the ELBO, `alpha_pip` or `joint` rule, with separate `alpha_tol` and
`pip_tol` (default `1e-4`), and can raise `min_iter` or lower the cycle lag.
For `R_finite`, `R_mismatch` and multi-panel fits it must keep the default
`min_iter` and cycle lag, and with `R_finite` or `R_mismatch` it can select only
the ELBO rule; other settings raise `ValueError`.
[Convergence and refinement](post_analysis.md#convergence-diagnostics) describes
these rules and the recorded diagnostics.

### Defaults by entry point

| Argument | `susie` | `susie_suff_stat` | `susie_rss` | `susie_rss_lambda` |
| --- | --- | --- | --- | --- |
| `L` | 10 | 10 | 10 | 10 |
| Initial prior variance | `scaled_prior_variance=0.2` | `scaled_prior_variance=0.2` | `scaled_prior_variance=0.2`; `prior_variance=50` without `n` | `prior_variance=50` |
| `estimate_residual_variance` | `True` | `True` | `False` | `False` |
| `estimate_residual_method` | `"MoM"` | `"MoM"` | `"MoM"` | `"MLE"` only |
| `residual_variance_lowerbound` | $\widehat{\operatorname{var}}(y)/10^4$ | 0 | 0 | 0 |
| `standardize` | `True` | `True` | `True`; off without `n` | off (no argument) |
| `max_iter` | 100 | 100 | 50 | 50 |
| `tol` | `None`: `1e-4`, or `1e-6` with NIG | `1e-4` | `1e-4` | `1e-3` |
| Stopping rule | ELBO | ELBO | ELBO; PIP as listed above | ELBO |
| `check_prior` | off (no argument) | `False` | `True` | `True` |

All four use `estimate_prior_variance=True`, `estimate_prior_method="optim"`,
`null_weight=0`, `prior_tol=1e-9`,
`check_null_threshold=0`, `coverage=0.95` and `min_abs_corr=0.5`. `L` is capped
at the number of columns, including any null column, except that
`susie_rss_lambda` caps it at the retained variant count after MAF filtering,
excluding the null column. For RSS-lambda, explicit `L=None` remains an alias
for 10, and explicit positive integer values are capped at that same count.
Passing `tol=None` to
`susie_suff_stat` or `susie_rss` also selects `1e-6` with NIG and `1e-4`
otherwise. `susie_auto` runs staged `susie` fits with `init_tol=1` and
`max_iter=100` per stage, then a final fit with `tol=0.01`.

## From summaries to working sufficient statistics

The Gaussian likelihood depends on the data through $\mathbf Q$, $\mathbf g$,
$s_y$ and $n$. `susie_suff_stat` accepts these quantities directly.
`susie_rss` constructs them from association summary statistics and LD, allowing
the same regression model to be fitted when individual-level data are
unavailable. The construction depends on which summaries are supplied and
determines the units of the returned effects. The quantities below are
constructed before the optional predictor standardization described above.

### Known sample size and Wald z statistics

With z-scores and a known sample size $n$, the default `z_method="wald"`
converts the marginal Wald statistics to the regression scale using

$$
\widetilde z_j=z_j\sqrt{\frac{n-1}{z_j^2+n-2}}.
$$

This adjustment accounts for the residual variance used in the denominator of
a univariate Wald statistic. When `var_y` is omitted, the response is assigned
unit sample variance and the working quantities are

$$
\begin{aligned}
\mathbf Q&=(n-1)\mathbf R,\\
\mathbf g&=\sqrt{n-1}\,\widetilde{\mathbf z},\\
s_y&=n-1.
\end{aligned}
$$

With `z_method="score"`, the supplied statistics are used directly,
$\widetilde z_j=z_j$. Supplying `bhat` and `shat` without `var_y` also uses
this construction after forming $z_j=\widehat\beta_j/\operatorname{SE}_j$.
The returned coefficients are therefore on the standardized working scale;
the units of the original effect estimates are not recovered from their ratio.

### Effect estimates, standard errors and phenotype variance

Supplying `bhat`, `shat`, `n` and `var_y` together provides the information
needed to reconstruct the original regression scale. Write
$v_y=\mathtt{var\_y}$ and $A_j=(n-1)/(z_j^2+n-2)$ for Wald statistics
($A_j=1$ for score statistics). The reconstructed predictor sum of squares is

$$
h_j=\frac{v_y A_j}{\operatorname{SE}_j^2}.
$$

The sufficient statistics are then

$$
\begin{aligned}
Q_{jk}&=\sqrt{h_j}\,R_{jk}\sqrt{h_k},\\
g_j&=\frac{\widetilde z_j\sqrt{A_j}\,v_y}{\operatorname{SE}_j},\\
s_y&=(n-1)v_y.
\end{aligned}
$$

prusie averages the reconstructed matrix with its transpose to remove rounding
asymmetry. If predictor standardization is enabled, it uses the reconstructed
diagonal entries. `coef` reverses this standardization and returns effects per
original predictor unit in the original phenotype units, under the supplied
marginal-regression assumptions. `mu` and `mu2` remain on the working predictor
scale used during fitting.

Supplying **z, `var_y` and n** without `bhat` and `shat` uses a different
construction:

$$
\begin{aligned}
\mathbf Q&=(n-1)\mathbf R,\\
\mathbf g&=\sqrt{n-1}\,\widetilde{\mathbf z},\\
s_y&=(n-1)v_y.
\end{aligned}
$$

Here `var_y` changes the response sum of squares and the initial residual and
prior variances. It does not reconstruct predictor scales or rescale
$\mathbf g$ by $\sqrt{v_y}$. To obtain coefficients on the original regression
scale, supply the effect estimates and standard errors along with `n` and
`var_y`.

### Missing sample size

Without $n$, prusie uses a large-sample likelihood for noncentrality parameters.
The working quantities are

$$
\begin{aligned}
\mathbf Q&=\mathbf R,\\
\mathbf g&=\mathbf z,\\
s_y&=1.
\end{aligned}
$$

The internal calculation uses $n=2$ as a placeholder for this likelihood.
Predictor standardization is disabled, and `prior_variance` sets the initial
effect variance (default 50). If `bhat` and `shat` are supplied, their ratio
provides the z-scores. A supplied `var_y` is validated and recorded but does not
change these quantities. Returned coefficients describe noncentrality effects;
original genotype and phenotype units cannot be recovered through this route.
prusie issues a warning when the sample size is omitted.

### Interpreting the LD approximation

$\mathbf R$ must contain signed correlations for the same variants, in the same
order and with the same effect-allele orientation as the association summaries.
[Inputs](inputs.md#variant-order-and-alleles) describes the required alignment.
With LD from the study samples and matching regression preprocessing, the
reconstruction can recover the individual-level regression quantities. With an
external LD panel, it is an approximation whose accuracy depends on ancestry,
reference sample size and data preparation.

Residual-variance estimation relies on LD consistent with the association
study. Sampling noise or mismatch in a reference panel can affect that estimate.
`susie_rss` defaults to `estimate_residual_variance=False` and warns when
estimation is requested.

For case-control GWAS, RSS applies a Gaussian approximation to the association
statistics. Quantitative eQTL summaries use their reported regression scale and
sample size.

## PIP and credible sets

The marginal posterior inclusion probability (PIP) summarizes evidence for a
variant across the single-effect components. Let $\mathcal A$ contain the
components with $V_\ell>\mathtt{prior\_tol}$. Under the variational approximation,
the probability that at least one active component selects variant $j$ is

$$
\mathrm{PIP}_j=1-\prod_{\ell\in\mathcal A}(1-\alpha_{\ell j}).
$$

The product is the probability that all active components select other
variants. Thus, $\alpha_{\ell j}$ describes the location of one component,
while $\mathrm{PIP}_j$ combines evidence across components. Each row of
$\alpha$ sums to one; PIPs can sum to more than one because several variants
may contribute to the trait.

A credible set $C_\ell$ summarizes uncertainty about the location of component
$\ell$. Variants are ranked by decreasing $\alpha_{\ell j}$ and added until
their cumulative posterior probability reaches `coverage` (default 0.95):

$$
q_\ell(\gamma_\ell\in C_\ell)
=\sum_{j\in C_\ell}\alpha_{\ell j}
\geq\mathtt{coverage}.
$$

Adding the last variant can take the accumulated probability above the target.
The returned posterior mass records this attained value. It describes the
selected set under the fitted model; repeated-sampling coverage describes how
often sets contain a true effect across repeated datasets and is assessed
through calibration.

Credible sets are also screened for component activity, duplicate membership
and LD purity. Purity summarizes the absolute pairwise correlations among set
members. By default, the minimum absolute correlation must be at least 0.5
(`min_abs_corr=0.5`). A set of strongly correlated variants can have high
posterior mass while leaving uncertainty about which member carries the
effect. [Results](results.md#credible-set-fields) describes the activity
thresholds, ordering, filtering options and returned component IDs.

## Additional model options

A fixed mixture prior replaces a component's single Gaussian variance by

$$
p(a_\ell)=\sum_{k=1}^{K}w_k N(a_\ell;0,v_k),\qquad
w_k\geq0,\quad\sum_k w_k=1.
$$

`prior_variance_grid` supplies positive $v_k$ and `mixture_weights` supplies
$w_k$ (equal weights by default). The SER integrates over the grid and returns
posterior mixture moments; it does not optimize one scalar variance in place of
the mixture, and the returned `V` is the weighted mean variance $\sum_k w_kv_k$.

`estimate_residual_method="NIG"` uses the conjugate Normal–Inverse-Gamma
formulation, with $\sigma^2\sim\operatorname{IG}(\alpha_0/2,\beta_0/2)$ and
effect variance proportional to $\sigma^2$. Both hyperparameters must be
positive; their default is $1/\sqrt n$. This integrates residual variance
instead of the ordinary plug-in Gaussian SER. With NIG, residual variance is
always estimated, an estimated prior variance uses `EM`, `scaled_prior_variance`
is not multiplied by $s_y/(n-1)$, and the returned `V` is multiplied by each
component's residual-variance estimate. `susie_rss` requires `n` for NIG. When
combined with a fixed mixture prior, Gaussian mixture SER takes precedence,
while NIG still controls the outer residual-variance updates. For NIG, the
residual update is the inverse-gamma posterior mean,
$\sigma^2\leftarrow(\beta_0+\mathrm{ER2})/(\alpha_0+n-2)$. `"MoM"` (the default)
and `"MLE"` retain the Gaussian SER and both use
$\sigma^2\leftarrow\mathrm{ER2}/n$; `susie_rss_lambda` accepts only `"MLE"` and
maximizes its likelihood over $\sigma^2$ numerically. Residual lower and upper
bounds constrain these updates.

`slot_prior` accepts `slot_prior_betabinom()` or `slot_prior_poisson(C)`, which
place hierarchical priors on effect activity. The posterior activity probability
$c_\ell$ weights each component's contribution, so its fitted mean becomes
$c_\ell\boldsymbol\alpha_\ell\odot\boldsymbol\mu_\ell$. These models, and NIG
with more than one effect, use the PIP stopping rule because the ordinary
Gaussian ELBO is not their fitting objective.

For summary statistics, `z_method="score"` uses the supplied statistics without
the finite-sample Wald adjustment. `R_finite` gives the sample size of a finite
LD reference (`True` uses the rows of reference `X`), and `R_mismatch`
(`"eb"`, `"eb_mix"`, `"eb_ser_init"`, `"eb_force_init"` or `"eb_no_init"`)
estimates reference mismatch, with controls from `susie_rss_control`.
`susie_rss_lambda` instead fits the covariance $\sigma^2\mathbf R+\lambda\mathbf I$,
with $\lambda$ given or estimated through `lambda_`. These options change the
likelihood and should be selected for the analysis design, rather than as
numerical remedies for misaligned alleles or variants.

`susie_auto` doubles $L$ from `L_init=1` until some estimated $V_\ell$ is zero
or $L$ exceeds `L_max=512`. `L_greedy` grows $L$ in steps of `L_greedy`,
warm-starting each fit from the previous one, until the smallest component log
Bayes factor falls below `greedy_lbf_cutoff` (default 0.1) or the requested $L$
is reached. `susie_trendfilter` uses cumulative predictor columns applied by an
implicit operator. SuSiE-inf, SuSiE-ash and Mr.ASH are not implemented;
`unmappable_effects` accepts only `"none"`.

## Computational compatibility

The numerical reference for the core fitting routines is susieR 0.16.6,
commit `8e56a8e038e989856d106d9ca5175cc664fea9d2`. prusie implements the
Gaussian SuSiE model described above using float64 arithmetic in Python and
Rust. Its [computational optimizations](optimization.md) reuse intermediate
quantities, reduce matrix traversals and regroup arithmetic without changing
the statistical model. Floating-point summation order and math-library
implementations can differ from R, so agreement is assessed numerically;
bitwise equality is not required.

A comparison uses the same inputs, verified variant order and matched fitting
settings, including $L$, prior weights, variance estimation, standardization
and stopping rules. The current acceptance criterion applies to the complete
final matrix of component posterior probabilities, $\boldsymbol\alpha$.
Let $\alpha^{\mathrm P}_{\ell j}$ and $\alpha^{\mathrm R}_{\ell j}$ be the
values returned by prusie and susieR for component $\ell$ and variant $j$.
Their **absolute error** is

$$
e^{\mathrm{abs}}_{\ell j}
=\left|\alpha^{\mathrm P}_{\ell j}-\alpha^{\mathrm R}_{\ell j}\right|.
$$

The comparison passes the numerical criterion when

$$
\max_{\ell,j} e^{\mathrm{abs}}_{\ell j}\le10^{-5}.
$$

This uses `atol=1e-5` and `rtol=0`: every entry has the same absolute-error
allowance, independent of the magnitude of the reference value.

Both matrices must have the same shape, with component rows and variant columns
in their original order. The comparison includes inactive components and
components without credible sets; no rows are omitted or permuted to improve
agreement. Each alpha matrix must also contain finite probabilities in $[0,1]$,
with every row summing to one within `1e-10`.

Marginal PIPs, credible sets, posterior moments, variances, Bayes factors, ELBOs
and convergence metadata are recorded as additional diagnostics. They are not
required to be identical by this alpha criterion. In particular, small numerical
differences near a component-activity or credible-set threshold can change the
reported summaries. The fitting thresholds and stopping tolerances remain those
specified for the comparison.

The [numerical accuracy](numerical_accuracy.md) page reports the tested
configurations and recorded results. Agreement on those cases does not establish
agreement for every input, model option or platform. Refinement has additional
search controls and
[documented differences from susieR](post_analysis.md#refinement-compared-with-susier);
core-fit agreement alone does not establish refinement equivalence.

## Methods and implementation references

The model and variational fitting method follow
[Wang et al. (2020), *A simple new approach to variable selection in regression,
with application to genetic fine mapping*](https://doi.org/10.1111/rssb.12388).
The summary-statistic construction follows
[Zou et al. (2022), *Fine-mapping from summary data with the “Sum of Single
Effects” model*](https://doi.org/10.1371/journal.pgen.1010299).

For the executed path, read
[api.py](https://github.com/LucaJiang/prusie/blob/main/src/prusie/api.py)
(`susie_rss`, `susie_suff_stat`),
[individual.py](https://github.com/LucaJiang/prusie/blob/main/src/prusie/individual.py)
(`susie`),
[solver crate](https://github.com/LucaJiang/prusie/tree/main/rust/prusie-core/src)
(`fit_controlled` and `fit_operator` in `ibss.rs`, `single_effect` in `ser.rs`)
and
[posterior.py](https://github.com/LucaJiang/prusie/blob/main/src/prusie/posterior.py)
(`marginal_pip`, `credible_sets`).
The [reference mapping](implementation.md#reference-source-mapping) identifies
corresponding functions at the pinned susieR commit.
