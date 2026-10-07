# Statistical model

Fine-mapping asks which variants in a region contribute to a trait when nearby
variants are correlated. Several variants can therefore explain much the same
association signal. SuSiE represents multiple effects while retaining uncertainty
about the variant responsible for each one. prusie fits the Gaussian SuSiE model
from association summary statistics and signed LD, or from centered sufficient
statistics.

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
counted effect alleles. The residual variance is $\sigma^2$; component $\ell$
has nonzero-effect prior variance $V_\ell$, and $\pi_j$ is its prior probability
of choosing variant $j$. prusie uses the same variant weights for all components.

By default, predictors are standardized using their sample standard deviations
$d_j=\sqrt{Q_{jj}/(n-1)}$. Inference then uses
$\mathbf D^{-1}\mathbf Q\mathbf D^{-1}$ and $\mathbf D^{-1}\mathbf g$, where
$\mathbf D=\operatorname{diag}(d_1,\ldots,d_p)$, while the response scale stays
fixed. A zero predictor scale is replaced by one for this transformation.
Below, $\mathbf Q$ and $\mathbf g$ in the updates denote these **working**
sufficient statistics. Setting `standardize=False` retains the supplied
predictor scale. `coef` converts the posterior mean back by dividing by the
stored predictor scales. Centering and any covariate adjustment must already
be consistent across the caller's summaries; this API does not fit covariates.

## A sum of single effects

The sampling model is Gaussian regression:

$$
\begin{aligned}
\mathbf y&=\mathbf X\mathbf b+\boldsymbol\varepsilon,
&\boldsymbol\varepsilon&\sim N(\mathbf0,\sigma^2\mathbf I_n),\\
\mathbf b&=\sum_{\ell=1}^{L}\mathbf b_\ell.
\end{aligned}
$$

Each component chooses one variant and gives it a Gaussian effect:

$$
\begin{aligned}
\gamma_\ell&\sim\operatorname{Categorical}(\pi_1,\ldots,\pi_p),\\
a_\ell&\sim N(0,V_\ell),
&\mathbf b_\ell&=a_\ell\mathbf e_{\gamma_\ell}.
\end{aligned}
$$

Here $\mathbf e_j$ is the vector with a one at position $j$ and zeros elsewhere.
The location $\gamma_\ell$ describes *which* variant carries the component;
$a_\ell$ describes its effect size. Uniform weights, $\pi_j=1/p$, are the default.
`prior_weights` changes these probabilities. `scaled_prior_variance` initializes
$V_\ell$ relative to $s_y/(n-1)$.

$L$ bounds the number of component effects. Multiple components can select the
same variant, and an estimated $V_\ell=0$ removes that component's nonzero
effect. The number of returned credible sets additionally depends on posterior
mass, LD purity and deduplication. Thus $L$, the number of distinct causal
variants, and the number of reported sets describe different quantities.

## Component posteriors and effect means

SuSiE fits a variational approximation that factorizes between components:

$$
q(\mathbf b_1,\ldots,\mathbf b_L)=\prod_{\ell=1}^{L}q_\ell(\mathbf b_\ell).
$$

Within a component, uncertainty about the chosen variant and its effect remains
coupled. The returned arrays represent

$$
\begin{aligned}
\alpha_{\ell j}&=q_\ell(\gamma_\ell=j),\\
\mu_{\ell j}&=\mathbb E_q[a_\ell\mid\gamma_\ell=j],\\
M^{(2)}_{\ell j}&=\mathbb E_q[a_\ell^2\mid\gamma_\ell=j]
=\mu_{\ell j}^2+\tau_{\ell j}^2.
\end{aligned}
$$

`alpha`, `mu` and `mu2` correspond to $\alpha$, $\mu$ and $M^{(2)}$.
In particular, `mu2` includes the conditional posterior variance
$\tau_{\ell j}^2$; it is neither the square of `mu` alone nor a variance array.
The component's marginal mean is
$\mathbf m_\ell=\boldsymbol\alpha_\ell\odot\boldsymbol\mu_\ell$ and the total
mean is $\mathbf m=\sum_\ell\mathbf m_\ell$. A large conditional mean need not
imply a large marginal mean if that variant has little component probability.
`coef` reports $m_j/d_j$ on the input predictor scale. The precise response and
coefficient units for each RSS input route are described below.

## One single-effect regression

To update component $\ell$, hold the other component means fixed and form

$$
\begin{aligned}
\mathbf r_\ell&=\mathbf g-\mathbf Q\sum_{k\ne\ell}\mathbf m_k,\\
\widehat b_{\ell j}&=r_{\ell j}/Q_{jj},
&s_j^2&=\sigma^2/Q_{jj}.
\end{aligned}
$$

$\mathbf r_\ell$ is a **length-$p$ crossproduct residual**. It is the
crossproduct of $\mathbf X$ with an individual-level residual, rather than the
length-$n$ residual itself. For $Q_{jj}>0$, a Gaussian single-effect regression
(SER) compares the effect prior with a zero effect at each candidate variant:

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

The posterior mean shrinks the marginal estimate toward zero, most strongly
when its sampling variance is large relative to the prior variance. Prior
weights affect the probability of a location; uncertainty can remain spread
across correlated variants with similar evidence. The formulas give the
ordinary positive-variance model. The implementation's weight smoothing and
zero-information conventions are collected in [Inputs](inputs.md#priors-and-zero-information-columns).
`lbf_variable` stores the variant log Bayes factors, before variant prior
weights are added; `lbf` stores the component's log weighted evidence.

## Sequential IBSS and variance estimation

Iterative Bayesian stepwise selection (IBSS) repeatedly applies SER. At
initialization, the component means are zero, each alpha row is uniform, and
the variances use the supplied initial values. Within a sweep, later components
use the newly updated means of earlier components:

```text
initialize component means, variances and their matrix products
for each sweep, up to max_iter:
    for each component in order:
        remove its old contribution from the fitted crossproduct
        form its crossproduct residual
        select V and compute the SER posterior (EM updates V after the posterior)
        store the component evidence, KL and new matrix product
    evaluate expected residual sum of squares and ELBO
    test convergence after the first sweep
    if converged: stop
    if requested: update residual variance for the next sweep
trim components below prior_tol and construct summaries
```

For `estimate_prior_method="optim"`, the component update maximizes the SER
log marginal likelihood $\log\sum_j\pi_j\mathrm{BF}_{\ell j}(V)$ using a Brent
search over log variance, with the implementation's smoothed weights. The
candidate is compared with the previous variance, then with $V=0$.
`check_null_threshold` requires the selected variance to improve on the null
log likelihood by more than that threshold. `simple` performs this null
comparison at the existing $V$ without the continuous search.

`EM` first computes the posterior at the current $V_\ell$, then sets
$V_\ell\leftarrow\sum_j\alpha_{\ell j}M^{(2)}_{\ell j}$ for the next update.
It does not apply the optim/simple null Bayes-factor comparison. The stored
component evidence and KL come from that just-computed posterior. With
`estimate_prior_variance=False`, SER uses fixed initialized $V$ and bypasses
both estimation and the null comparison; final low-variance trimming still
applies.

The variational objective balances expected data fit against departure from
the prior:

$$
\mathcal L(q)=\mathbb E_q[\log p(\mathbf y\mid\mathbf b,\sigma^2)]
-\sum_{\ell=1}^{L}\operatorname{KL}\{q_\ell(\mathbf b_\ell)\|p_\ell(\mathbf b_\ell)\}.
$$

The Gaussian likelihood needs the expected residual sum of squares, including
posterior effect uncertainty. [Implementation](implementation.md#expected-residual-sum-of-squares)
derives this from `mu2` and the retained component products. The objective is
stored at the current $\sigma^2$ **before** a possible residual-variance update.
If stopping has not occurred and estimation is enabled, the update is
$\sigma^2\leftarrow\mathrm{ER2}/n$, including after the last allowed sweep.
Consequently a fit that stops at the iteration limit can return a newer
`sigma2` than the value used for its last stored ELBO. A converged fit skips
that update. Final trimming does not recompute the ELBO or fitted crossproduct.
[Results](results.md#iterations-and-finalization) defines the stopping rule,
exceptional fallback and returned states in one place.

## From summaries to working sufficient statistics

With actual centered sufficient statistics, the Gaussian likelihood depends
on the data through $\mathbf Q$, $\mathbf g$, $s_y$ and $n$; an individual-level
matrix need not be retained. `susie_suff_stat` uses these quantities directly.
RSS constructs working crossproducts from marginal associations and LD.
When the LD and summaries describe the same data and preprocessing, this can
recover the regression quantities. External LD instead supplies an approximation
to the study's correlation structure.

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

With **z and `var_y` alone**, the current interface keeps
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
