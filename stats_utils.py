import numpy as np
from scipy import stats

def permutationTest(x, y, n_perm=10000, rng=None):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    diffs = x - y
    diffs = diffs[~np.isnan(diffs)]
    n = len(diffs)
    observed = diffs.mean()

    if rng is None:
        rng = np.random.default_rng()
    signs = rng.choice([-1, 1], size=(n_perm, n))
    perm_stats = (signs * diffs).mean(axis=1)
    pval = (np.sum(np.abs(perm_stats) >= np.abs(observed)) + 1) / (n_perm + 1)

    return observed, pval

def _to_radians(arr, units):
    """Convert a finite-value-only array to radians, given its native units."""
    if units == 'hours':
        return arr * (2 * np.pi / 24.0)
    elif units == 'radians':
        return arr
    else:
        raise ValueError("units must be 'hours' or 'radians'")


def _from_radians(arr, units):
    """Inverse of _to_radians — convert radians back to the caller's native units."""
    if units == 'hours':
        return arr * (24.0 / (2 * np.pi))
    elif units == 'radians':
        return arr
    else:
        raise ValueError("units must be 'hours' or 'radians'")


def mardia_watson_wheeler(*groups, units='hours'):
    """
    Mardia-Watson-Wheeler omnibus test (harmonics m=1,2) for circular data.
    Returns chi-square W and p-value (df = 2*(k-1)).

    Parameters
    ----------
    *groups : array-like
        One array per group, in the units specified by `units`.
    units : {'hours', 'radians'}, default 'hours'
        Whether the input data are hours on a 0-24 clock or radians on a
        0-2*pi circle.
    """
    theta_groups = []
    for g in groups:
        arr = np.asarray(g, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            raise ValueError("All groups must contain at least one finite value.")
        theta_groups.append(_to_radians(arr, units))

    k = len(theta_groups)
    if k < 2:
        raise ValueError("Provide at least two groups.")
    n_g = np.array([len(g) for g in theta_groups], dtype=int)
    n = int(np.sum(n_g))

    W = 0.0
    for m in (1, 2):
        C_mg = np.array([np.sum(np.cos(m * g)) for g in theta_groups], dtype=float)
        S_mg = np.array([np.sum(np.sin(m * g)) for g in theta_groups], dtype=float)
        group_term = np.sum((C_mg**2 + S_mg**2) / n_g)

        C_m = np.sum(C_mg)
        S_m = np.sum(S_mg)
        pooled_term = (C_m**2 + S_m**2) / n

        W += 2.0 * (group_term - pooled_term)

    dof = 2 * (k - 1)
    p_chi = 1.0 - stats.chi2.cdf(W, dof)
    return float(W), float(p_chi), dof


def mww_permutation_pvalue(*groups, units='hours', B=10000, random_state=None):
    """
    Permutation p-value for MWW, shuffling labels while preserving group sizes.
    Returns (W_obs, p_perm, df, p_chi) where p_perm is the permutation p-value.

    Parameters
    ----------
    *groups : array-like
        One array per group, in the units specified by `units`.
    units : {'hours', 'radians'}, default 'hours'
        Whether the input data are hours on a 0-24 clock or radians on a
        0-2*pi circle.
    """
    rng = np.random.default_rng(random_state)

    # Compute observed W
    W_obs, p_chi, df = mardia_watson_wheeler(*groups, units=units)

    # Prepare pooled data (in radians) and group sizes
    theta_groups = []
    sizes = []
    for g in groups:
        arr = np.asarray(g, dtype=float)
        arr = arr[np.isfinite(arr)]
        theta_groups.append(_to_radians(arr, units))
        sizes.append(len(arr))
    sizes = np.array(sizes, dtype=int)

    pooled = np.concatenate(theta_groups)
    n = pooled.size
    idx = np.arange(n)

    def W_from_partition(parts):
        # parts is a list of index arrays for each group
        split_groups = [pooled[p] for p in parts]
        # Convert back to the caller's native units, since mardia_watson_wheeler
        # will re-convert to radians internally according to `units`.
        native_groups = [_from_radians(g, units) for g in split_groups]
        W, _, _ = mardia_watson_wheeler(*native_groups, units=units)
        return W

    # Observed W already computed
    count_ge = 0
    for _ in range(B):
        rng.shuffle(idx)
        # Build partition with preserved sizes
        parts = []
        start = 0
        for s in sizes:
            parts.append(idx[start:start + s])
            start += s
        W_perm = W_from_partition(parts)
        if W_perm >= W_obs:
            count_ge += 1

    p_perm = (count_ge + 1) / (B + 1)
    return float(W_obs), float(p_perm), df, float(p_chi)
