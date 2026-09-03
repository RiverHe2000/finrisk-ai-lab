# Methodology

Formulas implemented in `ifrs9_pd`, in pipeline order. Notation: $y_i \in \{0,1\}$
is the 12-month default flag of loan-snapshot $i$, $p_i$ its predicted PD,
$\Phi$ the standard normal CDF.

## 1. Weight of evidence and information value (`features/binning.py`)

For bin $k$ of a feature with $g_k$ non-defaults and $b_k$ defaults out of $G$
and $B$ in total, with Laplace smoothing $a = 1/2$ and $K$ bins:

$$
\mathrm{WoE}_k = \ln\frac{(g_k + a)/(G + Ka)}{(b_k + a)/(B + Ka)}, \qquad
\mathrm{IV} = \sum_k \left(\frac{g_k + a}{G + Ka} - \frac{b_k + a}{B + Ka}\right)\mathrm{WoE}_k .
$$

Positive WoE means lower risk than the portfolio average. Numeric features start
from $n$ quantile bins; adjacent bins are merged until every bin holds at least
`min_bin_share` of observations and event rates are monotonic in bin order
(direction taken from the count-weighted covariance of bin index and event
rate). Missing values form their own bin. Categorical features get one bin per
category with categories under `min_bin_share` pooled into `other`; unseen
categories at scoring time map to the `other` WoE.

Feature selection keeps features with $\mathrm{IV} \ge$ `min_iv` (0.02) and,
optionally, below `max_iv` as a leakage guard.

## 2. Logistic scorecard and points scaling (`model/scorecard.py`)

$$
\operatorname{logit} p_i = \alpha + \sum_j \beta_j\,\mathrm{WoE}_j(x_{ij}) + \delta ,
$$

with $\beta$ from an L2-regularised logistic regression and $\delta$ the
calibration shift of §3. With $\text{factor} = \mathrm{PDO}/\ln 2$ and
$\text{offset} = \text{base\_score} - \text{factor}\cdot\ln(\text{base\_odds})$,

$$
\text{score} = \text{offset} + \text{factor}\cdot\ln\frac{1-p}{p}
= \sum_j \underbrace{\Big[-\text{factor}\,\beta_j\,\mathrm{WoE}_j + \tfrac{\text{offset} - \text{factor}(\alpha+\delta)}{J}\Big]}_{\text{points}_j} .
$$

Defaults: PDO 20, 600 points at 50:1 odds. The master scale has $n$ grades whose
$n-1$ interior boundaries are geometrically spaced between 0.03% and 30%, so
each grade multiplies PD by roughly 2.4.

## 3. Central-tendency calibration and factor loading (`model/calibration.py`)

The intercept shift $\delta$ solves

$$
\frac{1}{N}\sum_i \mathrm{PIT}\!\left(\sigma(\operatorname{logit} p_i^{\text{raw}} + \delta),\ \lambda z_i\right) = \overline{\mathrm{DR}}_{\text{dev}},
$$

i.e. the anchor is imposed on the PIT PDs that are compared with observed
defaults. The factor loading $\lambda$ is chosen by maximising the Bernoulli
log-likelihood $\sum_i y_i \ln p_i + (1-y_i)\ln(1-p_i)$ over $\lambda \in [0, 3]$,
re-solving $\delta$ for each candidate.

## 4. Vasicek point-in-time / through-the-cycle conversion

Single-factor model with asset correlation $\rho$ (0.15 for residential
mortgages, Basel II) and systematic factor $z$:

$$
\mathrm{PD}_{\text{PIT}}(z) = \Phi\!\left(\frac{\Phi^{-1}(\mathrm{PD}_{\text{TTC}}) - \sqrt{\rho}\,z}{\sqrt{1-\rho}}\right), \qquad
\mathrm{PD}_{\text{TTC}} = \Phi\!\left(\sqrt{1-\rho}\,\Phi^{-1}(\mathrm{PD}_{\text{PIT}}) + \sqrt{\rho}\,z\right).
$$

$z$ is proxied by the negative standardised unemployment gap,
$z_t = -(u_t - \bar u)/\sigma_u$ (optionally the standardised 12-month change),
scaled by the fitted loading $\lambda$. Negative $z$ is adverse. Note that
$\mathrm{PD}_{\text{PIT}}(0) < \mathrm{PD}_{\text{TTC}}$ by Jensen's inequality;
the unconditional PD is the expectation over $z$.

## 5. Lifetime term structure (`model/term_structure.py`)

For scenario $s$ with initial factor $z_0 + \text{shift}_s$ and monthly mean
reversion $\kappa$:

$$
z_t^s = (z_0 + \text{shift}_s)(1-\kappa)^{t-1}, \qquad
\mathrm{PD12}_t^s = \Phi\!\left(\frac{\Phi^{-1}(\mathrm{PD12}_{\text{TTC}}) - \sqrt{\rho}\, z_t^s}{\sqrt{1-\rho}}\right),
$$

$$
h_t^s = m(a + t)\left[1 - (1-\mathrm{PD12}_t^s)^{1/12}\right],
$$

where $a$ is the loan's current age and the seasoning multiplier
$m(u) \propto 1 + (\mu - 1)\,\tfrac{u}{p}\,e^{1 - u/p}$ (peak $\mu$ = 1.4 at
$p$ = 24 months) is normalised per loan so that $\tfrac{1}{12}\sum_{t=1}^{12} m(a+t) = 1$,
preserving the scorecard's 12-month PD. Then

$$
S_t = \prod_{k \le t}(1 - h_k), \qquad
\mathrm{PD}^{\text{marg}}_t = S_{t-1} h_t, \qquad
\mathrm{PD}^{\text{cum}}_t = 1 - S_t ,
$$

and the probability-weighted curve is $\sum_s w_s \mathrm{PD}^{\text{marg},s}_t$
(base 0.50 / upside 0.25 / downside 0.25 with shifts 0 / +1.0 / -1.5). The
transform is applied at the annual horizon rather than to monthly hazards so the
factor sensitivity matches the 12-month conversion of §4.

## 6. Staging (`model/staging.py`)

$$
\text{Stage} =
\begin{cases}
3 & \text{dpd} \ge 90 \\
2 & \text{dpd} \ge 30 \ \text{or}\ \left(\dfrac{\mathrm{PD}_{\text{now}}}{\mathrm{PD}_{\text{orig}}} \ge 2 \ \text{and}\ \mathrm{PD}_{\text{now}} - \mathrm{PD}_{\text{orig}} \ge 50\,\text{bp}\right) \\
1 & \text{otherwise}
\end{cases}
$$

$\mathrm{PD}_{\text{orig}}$ is the age-matched origination PD: the loan re-scored
with delinquency fields reset and the macro factor at origination.

## 7. Expected credit loss (`model/ecl.py`)

With monthly discount factors $DF_t = (1+r)^{-t/12}$ and an exposure profile
$\mathrm{EAD}_t = \mathrm{EAD}\cdot\max\!\left(0, \tfrac{T - t + 1}{T}\right)$ for
remaining term $T$:

$$
\mathrm{ECL}_{12m} = \sum_{t=1}^{12} \mathrm{PD}^{\text{marg}}_t\,\mathrm{LGD}\,\mathrm{EAD}_t\,DF_t, \qquad
\mathrm{ECL}_{\text{life}} = \sum_{t=1}^{H} \mathrm{PD}^{\text{marg}}_t\,\mathrm{LGD}\,\mathrm{EAD}_t\,DF_t, \qquad
\mathrm{ECL}_{\text{stage 3}} = \mathrm{EAD}\cdot\mathrm{LGD}.
$$

Stage 1 carries $\mathrm{ECL}_{12m}$, Stage 2 $\mathrm{ECL}_{\text{life}}$
(with $H$ = 60 months), Stage 3 the credit-impaired amount.

## 8. Discrimination (`validation/discrimination.py`)

AUC via the Mann-Whitney statistic with average ranks for ties,
$\mathrm{Gini} = 2\,\mathrm{AUC} - 1$, and

$$
\mathrm{KS} = \max_s \left| F_{\text{bad}}(s) - F_{\text{good}}(s) \right| .
$$

Confidence intervals are percentile bootstraps (300 resamples, seeded). Rank
ordering requires observed default rates non-decreasing across grades, ignoring
pairs where either grade holds fewer than five defaults.

## 9. Calibration (`validation/calibration.py`)

Hosmer-Lemeshow over $g$ PD-deciles with $O_k$ observed and $E_k$ expected
defaults out of $n_k$:

$$
\mathrm{HL} = \sum_{k=1}^{g} \frac{(O_k - E_k)^2}{E_k\,(1 - E_k/n_k)} \sim \chi^2_{g-2}.
$$

Binomial test per grade: two-sided exact test of $d_k$ defaults out of $n_k$
against the mean predicted PD. Jeffreys test per grade: with posterior
$\mathrm{DR} \sim \mathrm{Beta}(d_k + \tfrac12,\ n_k - d_k + \tfrac12)$ the
reported p-value is $P(\mathrm{DR} \le \overline{\mathrm{PD}}_k)$; small values
flag under-estimation. Brier score $= \tfrac{1}{N}\sum_i (p_i - y_i)^2$.

## 10. Stability (`validation/stability.py`)

With expected (development) and actual (recent) shares $e_k$, $a_k$ over $K$
quantile bins of the development distribution (empty shares floored at
$10^{-6}$):

$$
\mathrm{PSI} = \sum_{k=1}^{K} (a_k - e_k)\,\ln\frac{a_k}{e_k}.
$$

The characteristic stability index applies the same formula to each input
feature (one bin per category for categorical features, a separate missing bin).

## 11. Synthetic data-generating process (`data/synthetic.py`)

Borrower-level log-odds are linear in bureau score, LTV, DTI, utilisation,
arrears, days past due, interest-only flag, employment type and region, plus a
seasoning hump $0.4\,\tfrac{m}{24}\,e^{1 - m/24}$ in months on book. The true
PIT PD applies the Vasicek transform of §4 with $\rho = 0.15$ and
$z_t = -(u_t - 5)/2$, where $u_t$ is a smooth unemployment path with a
Gaussian stress bump peaking in June 2020. Labels are Bernoulli draws from the
true PIT PD; 1.5% of bureau scores and DTIs are then set missing.
