# Exact metric definitions

## Frozen states versus parameter linearization

Every checkpoint is actually loaded and fully forwarded on the same stored
prefix and continuation token IDs. No parameter Jacobian
\(J_{\theta_0}\Delta\theta\), parameter-space Fisher, gradient prediction, or
first-order approximation in \(\Delta\theta\) is used.

## Functional vector

At each frozen prediction position, \(S\) is the Initial student's top-128 token
set. Let \(p_0(i)\) be the Initial student's full-vocabulary-normalized
probability for \(i\in S\), \(m_0=\sum_{i\in S}p_0(i)\), and
\(q_0(i)=p_0(i)/m_0\).

For an actually forwarded finite endpoint \(\theta\),

\[
d_\theta(i)=\log p_\theta(i)-\log p_0(i),
\qquad
\bar d_\theta=\sum_{i\in S}q_0(i)d_\theta(i),
\]

and the implemented vector is

\[
v_\theta(i)=\sqrt{p_0(i)}\,[d_\theta(i)-\bar d_\theta].
\]

Thus:

- the raw quantity is a finite endpoint-minus-Initial log-probability change;
- it is not a probability difference;
- centering uses the Initial distribution conditioned on its frozen support;
- the multiplicative Fisher weight is the Initial student's unrenormalized
  probability \(p_0\) on each selected token;
- Teacher and endpoint probabilities do not define the metric weights.

`a`, cosine, and orthogonal residual are projections of these measured finite
endpoint vectors under that fixed Initial-output metric.

## Diagnostic KL

For every model separately, its probabilities on \(S\) are renormalized:

\[
q_\theta(i)=\frac{p_\theta(i)}{\sum_{k\in S}p_\theta(k)}.
\]

The diagnostic uses the teacher-first forward direction:

\[
D_{\mathrm{KL}}(q_T\Vert q_\theta)
=\sum_{i\in S}q_T(i)\log\frac{q_T(i)}{q_\theta(i)}.
\]

Therefore baseline KL is \(D_{\mathrm{KL}}(q_T\Vert q_0)\), remaining KL is
\(D_{\mathrm{KL}}(q_T\Vert q_D)\), and

\[
\text{gap closed}=1-\frac{\text{remaining KL}}{\text{baseline KL}}.
\]

This is an exact categorical KL on the conditional top-128 distributions, not
a quadratic Fisher approximation. It is an evaluation convention separate
from the training loss implementation.

## Probability outside top-128

Probability outside \(S\) is excluded from the conditional KL and from the
centered directional comparison. Each model's retained support mass
\(\sum_{i\in S}p_\theta(i)\) is reported separately by domain. The procedure
does not create an “other” bucket and does not charge escaped mass directly to
KL.

## Domain aggregation

For each requested domain, all position×token vector components are
concatenated. Inner products and squared norms are summed first, then one cosine
is computed.

For `all`, every position from every domain is concatenated before computing
the Gram matrix and cosine. It is not an arithmetic mean of domain cosines.

Consequences:

- positions are weighted by their Initial token probabilities through
  \(\sqrt{p_0}\);
- longer trajectories contribute more positions and therefore more total
  weight;
- domains with more scored positions contribute more to `all`;
- trajectories and domains are not equal-weighted in the point cosine.

Trajectory bootstrap, where reported, resamples whole trajectories and then
sums their inner-product/norm sufficient statistics before taking the cosine.
The source-balanced gap-closure confidence intervals instead resample tasks,
keeping all four trajectory origins from a task together.
