# Statistical model

Fine-mapping aims to identify the variants in a region that affect a trait
when nearby variants are correlated, so that several variants can explain much
the same association signal. SuSiE represents multiple effects and quantifies,
for each effect, the uncertainty about which variant carries it. prusie fits
the Gaussian SuSiE model to association summary statistics and signed LD, or to
centered sufficient statistics.

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

By default, predictors are standardized by their sample standard deviations
$d_j=\sqrt{Q_{jj}/(n-1)}$. Inference then uses
$\mathbf D^{-1}\mathbf Q\mathbf D^{-1}$ and $\mathbf D^{-1}\mathbf g$, where
$\mathbf D=\operatorname{diag}(d_1,\ldots,d_p)$; the response scale is not
changed. A zero predictor scale is replaced by one. In the updates below,
$\mathbf Q$ and $\mathbf g$ denote these **working** sufficient statistics.
With `standardize=False` the supplied predictor scale is used. `coef` returns
the posterior mean on the input scale by dividing by the stored predictor
scales. Centering and any covariate adjustment must already be consistent
across the supplied summaries; covariates are not fitted.

## A sum of single effects

The sampling model is Gaussian regression:

$$
\begin{aligned}
\mathbf y&=\mathbf X\mathbf b+\boldsymbol\varepsilon,
&\boldsymbol\varepsilon&\sim N(\mathbf0,\sigma^2\mathbf I_n),\\
\mathbf b&=\sum_{\ell=1}^{L}\mathbf b_\ell.
\end{aligned}
$$

Each single effect is located at one variant and has a Gaussian effect size:

$$
\begin{aligned}
\gamma_\ell&\sim\operatorname{Categorical}(\pi_1,\ldots,\pi_p),\\
a_\ell&\sim N(0,V_\ell),
&\mathbf b_\ell&=a_\ell\mathbf e_{\gamma_\ell}.
\end{aligned}
$$

Here $\mathbf e_j$ is the $j$th standard basis vector. The location
$\gamma_\ell$ indicates *which* variant carries the effect and $a_\ell$ its
size. The default prior is uniform, $\pi_j=1/p$; `prior_weights` sets other
values. `scaled_prior_variance` gives the initial $V_\ell$ relative to
$s_y/(n-1)$.

$L$ is an upper bound on the number of effects. Several single effects can be
located at the same variant, and an estimated $V_\ell=0$ removes an effect. The
number of reported credible sets depends in addition on posterior mass, LD
purity and the removal of duplicate sets. $L$, the number of causal variants
and the number of credible sets are therefore different quantities.

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

`alpha`, `mu` and `mu2` correspond to $\alpha$, $\mu$ and $M^{(2)}$; in the
API and results, single effects are also called components and are indexed by
`component_id`. `mu2` includes the conditional posterior variance
$\tau_{\ell j}^2$; it is neither the square of `mu` nor a variance. The
posterior mean of effect $\ell$ is
$\mathbf m_\ell=\boldsymbol\alpha_\ell\odot\boldsymbol\mu_\ell$ and the total
posterior mean is $\mathbf m=\sum_\ell\mathbf m_\ell$. A large conditional mean
does not imply a large posterior mean when the variant has a small inclusion
probability. `coef` reports $m_j/d_j$ on the input predictor scale. The
response and coefficient units of each RSS input route are given below.

## One single-effect regression

To update effect $\ell$, the posterior means of the other effects are held
fixed and

$$
\begin{aligned}
\mathbf r_\ell&=\mathbf g-\mathbf Q\sum_{k\ne\ell}\mathbf m_k,\\
\widehat b_{\ell j}&=r_{\ell j}/Q_{jj},
&s_j^2&=\sigma^2/Q_{jj}.
\end{aligned}
$$

$\mathbf r_\ell$ is a **crossproduct residual** of length $p$: the
crossproduct of $\mathbf X$ with the individual-level residual, not the
residual of length $n$ itself. For $Q_{jj}>0$, the Gaussian single-effect
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
\mu_{\ell j}&=\frac{V_\ell}{V_\ell+s_j^2}\widehat b_{\ell j},
&\tau_{\ell j}^2&=\frac{V_\ell s_j^2}{V_\ell+s_j^2}.
\end{aligned}
$$

The posterior mean shrinks the marginal estimate towards zero, the more so the
larger its sampling variance relative to the prior variance. For fixed prior
and residual variances, the prior weights affect the probability of each
location; posterior probability can remain spread over correlated variants
with similar evidence. These formulas describe
the case of a positive prior variance; the smoothing of the prior weights and
the treatment of columns without information are described under
[Inputs](inputs.md#priors-and-zero-information-columns). `lbf_variable` stores
the log Bayes factors of the variants before prior weighting; `lbf` stores the
log of the prior-weighted average Bayes factor of each effect.

## Sequential IBSS and variance estimation

Iterative Bayesian stepwise selection (IBSS) applies the SER to each effect in
turn. At a cold start, all posterior means are zero, each alpha row is uniform
and the variances take their initial values. Within a sweep, each effect is updated
using the current means of all other effects, including those already updated
in the same sweep:

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
    record diagnostics and check the requested stopping rule
    if success, an eligible cycle stop, or numerical failure: stop
    if requested: update residual variance for the next sweep
trim components below prior_tol and construct summaries
```

With a finite ELBO and the default rule, convergence can first be declared
after `min_iter` sweeps (two by default); repeated posterior states are only
recorded as diagnostics. [Convergence and refinement](post_analysis.md#convergence-diagnostics)
gives the exact objective condition and the exceptional stops.

With `estimate_prior_method="optim"`, each update maximizes the SER log marginal
likelihood $\log\sum_j\pi_j\mathrm{BF}_{\ell j}(V)$ over $\log V$ by Brent's
method, using the smoothed prior weights. The maximizer is compared first with
the current variance and then with $V=0$: $V$ is set to zero unless its log
marginal likelihood exceeds that at $V=0$ by more than `check_null_threshold`.
`simple` makes only this comparison with zero, at the current $V$.

`EM` first computes the posterior at the current $V_\ell$ and then sets
$V_\ell\leftarrow\sum_j\alpha_{\ell j}M^{(2)}_{\ell j}$ for the next update,
without the comparison with zero. The stored Bayes factors and KL refer to
that posterior. With `estimate_prior_variance=False`, the initial $V$ is kept
and neither estimation nor the comparison with zero is applied; the final
trimming of small variances still applies.

The variational objective (evidence lower bound, ELBO) is the expected log
likelihood minus the divergence of the approximation from the prior:

$$
\mathcal L(q)=\mathbb E_q[\log p(\mathbf y\mid\mathbf b,\sigma^2)]
-\sum_{\ell=1}^{L}\operatorname{KL}\{q_\ell(\mathbf b_\ell)\|p_\ell(\mathbf b_\ell)\}.
$$

The Gaussian expected log likelihood requires the expected residual sum of
squares (ER2), which accounts for posterior uncertainty in the effects;
[Implementation](implementation.md#expected-residual-sum-of-squares) derives it
from `mu2` and the stored matrix products. The ELBO is recorded at the current
$\sigma^2$, **before** any residual-variance update. If the fit has not stopped
and $\sigma^2$ is estimated, it is updated as $\sigma^2\leftarrow\mathrm{ER2}/n$,
also after the last permitted sweep. A fit that reaches the iteration limit can
therefore return a `sigma2` that differs from the value used for its last ELBO;
a converged fit omits this update. The final trimming recomputes neither the
ELBO nor the fitted crossproduct. [Results](results.md#iterations-and-finalization)
summarizes the stopping rule, the NaN fallback and the returned quantities.

## From summaries to working sufficient statistics

Given centered sufficient statistics, the Gaussian likelihood depends on the
data only through $\mathbf Q$, $\mathbf g$, $s_y$ and $n$, so individual-level
data are not needed; `susie_suff_stat` uses these quantities directly. SuSiE-RSS
constructs working sufficient statistics from marginal association statistics
and LD. When the LD and the summary statistics come from the same samples and
preprocessing, this can recover the regression quantities; an external LD
reference provides only an approximation to the correlation in the study.

### Known sample size and Wald z statistics

For the default response scale, prusie applies the finite-sample Wald/PVE
adjustment and constructs

$$
\begin{aligned}
\widetilde z_j&=z_j\sqrt{\frac{n-1}{z_j^2+n-2}},\\
\mathbf Q&=(n-1)\mathbf R,
&\mathbf g&=\sqrt{n-1}\,\widetilde{\mathbf z},
&s_y&=n-1.
\end{aligned}
$$

The adjustment relates a univariate Wald statistic, whose denominator uses a
residual variance estimate, to a crossproduct on the standardized-response
scale. Large Wald statistics shrink appreciably; for small effects and large
$n$, the factor is close to one. Supplying $n$ determines this conversion and
the crossproduct scale. For example, $n=100$ and $z=5$ give the following
script-checked values:

<!-- calculated:wald:start -->
The adjusted statistic is **4.485750** and the working crossproduct is **44.632647** (with $Q_{jj}=99$ for unit LD diagonal).
<!-- calculated:wald:end -->

`bhat/shat` without `var_y` follows this same route after forming
$z_j=\widehat\beta_j/\operatorname{SE}_j$. Its returned coefficients therefore
use the standardized working scale, even though the input estimates originally
had trait-specific units.

### Effect estimates, standard errors and phenotype variance

When `bhat`, `shat`, `n` and `var_y` are all supplied, write
$A_j=(n-1)/(z_j^2+n-2)$ and $v_y=\mathtt{var\_y}$. The reconstruction is

$$
\begin{aligned}
h_j&=\frac{v_y A_j}{\operatorname{SE}_j^2},\\
Q_{jk}&=\sqrt{h_j}\,R_{jk}\sqrt{h_k},\\
g_j&=\frac{\widetilde z_j\sqrt{A_j}\,v_y}{\operatorname{SE}_j},
&s_y&=(n-1)v_y.
\end{aligned}
$$

The constructed crossproduct is averaged with its transpose to remove rounding
asymmetry. Subsequent predictor standardization uses these reconstructed
diagonals. `coef` reverses that standardization and has the original predictor
and phenotype units under the supplied marginal-regression assumptions. `mu`
and `mu2` remain on the fitted predictor scale.

With **z, `var_y` and known n**, the current interface keeps
$\mathbf Q=(n-1)\mathbf R$ and
$\mathbf g=\sqrt{n-1}\,\widetilde{\mathbf z}$, but sets
$s_y=(n-1)v_y$. This matches the pinned constructor: it changes response
sum of squares, default residual variance and initial prior variance without
reconstructing predictor scales or multiplying $\mathbf g$ by
$\sqrt{v_y}$. These coefficients are defined by those working quantities;
use the effect-estimate route when original-scale reconstruction is needed.

### Missing sample size

When $n$ is omitted, prusie uses the noncentrality-parameter likelihood with
$\mathbf Q=\mathbf R$, $\mathbf g=\mathbf z$, $s_y=1$ and internal $n=2$.
Predictor standardization is disabled, and `prior_variance` initializes $V$
(default 50). `bhat/shat` supplies z in exactly the same way. A supplied
`var_y` is checked for validity and recorded but does not alter these working
quantities. `coef` then describes noncentrality effects, rather than effects
per original genotype/phenotype unit. A warning records this missing-n path.

### Interpreting the LD approximation

LD is signed correlation $r$, aligned to the same variants and effect alleles
as the summaries; [Inputs](inputs.md#variant-order-and-alleles) gives the required
transformations. Residual-variance estimation relies on in-sample LD consistent
with the summaries. With an external panel, RSS defaults to fixed variance so
that sampling noise or mismatch in LD is not interpreted as residual variance.
The approximation still depends on ancestry, sample size and data preparation.
For case-control GWAS, prusie uses the Gaussian summary approximation to the
association statistics; it does not interpret the binary phenotype as a fitted
individual-level Gaussian response. Quantitative eQTL summaries use their
reported regression scale and sample size.

## PIP and credible sets

For active components $\mathcal A$, the variational marginal inclusion
probability is

$$
\mathrm{PIP}_j=1-\prod_{\ell\in\mathcal A}(1-\alpha_{\ell j}).
$$

The product is the approximate probability that no active component selects
variant $j$. Each alpha row sums to one, whereas the sum of PIPs over variants
can exceed one because the model allows multiple effects. Consider two active
components with alpha rows $(0.8,0.15,0.05)$ and $(0.1,0.8,0.1)$:

<!-- calculated:pip:start -->
The resulting PIPs are **(0.82, 0.83, 0.145)**, whose sum is 1.795.
<!-- calculated:pip:end -->

A component credible set accumulates variants in decreasing alpha order until
its mass reaches `coverage`. For alpha $(0.50,0.48,0.02)$ and a requested mass
of 0.95, the two leading variants are selected:

<!-- calculated:cs:start -->
The actual posterior mass is **0.98**. If the two variants have signed correlation $r=-0.9$, the minimum, mean and median absolute pairwise correlation are all **0.9**, so the set passes <code>min_abs_corr=0.5</code>.
<!-- calculated:cs:end -->

The selected mass is the component posterior probability assigned to those
members; the requested mass is the selection target. Purity summarizes absolute
pairwise LD inside the selected set. Here either of two highly correlated
variants could carry the effect, so their joint mass is high even though
neither is individually well localized. Repeated-sampling coverage is the
frequency with which constructed sets contain a true effect over repeated
datasets; it is a calibration property, distinct from the mass computed for
this fit. [Results](results.md#credible-set-fields) describes activity thresholds,
stable ordering, deduplication and returned component IDs.

The arithmetic examples here and on the implementation page are generated and
checked by [model_examples.py](https://github.com/LucaJiang/prusie/blob/main/tools/model_examples.py).
They illustrate probability and linear algebra; the separately fitted
[500-variant toy example](example.md) demonstrates the complete API.

## Methods and implementation references

The model and variational fitting method follow
[Wang et al. (2020), *A simple new approach to variable selection in regression,
with application to genetic fine mapping*](https://doi.org/10.1111/rssb.12388).
The summary-statistic construction follows
[Zou et al. (2022), *Fine-mapping from summary data with the “Sum of Single
Effects” model*](https://doi.org/10.1371/journal.pgen.1010299).

For the executed path, read
[api.py](https://github.com/LucaJiang/prusie/blob/main/src/prusie/api.py)
(`susie_rss`, `_susie_suff_stat`),
[core.rs](https://github.com/LucaJiang/prusie/blob/main/rust/core.rs)
(`single_effect`, `fit_layout_operator`) and
[posterior.py](https://github.com/LucaJiang/prusie/blob/main/src/prusie/posterior.py).
The [reference mapping](implementation.md#reference-source-mapping) identifies
corresponding functions at the pinned susieR commit.
