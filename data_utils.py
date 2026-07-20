import numpy as np

def group_calendar_rolling(days, values, window_widths):
    n = len(days)
    if n == 0:
        return np.empty((0, len(window_widths)))
    order = np.argsort(days)
    days_s = days[order]
    vals_s = values[order]
    cumsum = np.concatenate(([0.0], np.cumsum(vals_s)))
    right_idx = np.arange(1, n + 1)
    out_sorted = np.full((n, len(window_widths)), np.nan, dtype=float)

    for j, w in enumerate(window_widths):
        left_edges = days_s - w
        left_idx = np.searchsorted(days_s, left_edges, side='right')
        counts = right_idx - left_idx
        sums = cumsum[right_idx] - cumsum[left_idx]
        mask = counts > 0
        out_sorted[mask, j] = sums[mask] / counts[mask]

    out = np.empty_like(out_sorted)
    out[order, :] = out_sorted
    return out
