import numpy as np
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle


def prep_series(df, timestamp_col='dummy_timestamp', val_col='lfp_left_OvER_interpolate_z_scored'):
    """Drop NaNs, sort, and convert timestamps to elapsed hours (float)."""
    d = df.dropna(subset=[val_col]).sort_values(timestamp_col)
    t0 = d[timestamp_col].iloc[0]
    t_hours = (d[timestamp_col] - t0).dt.total_seconds().values / 3600
    y = d[val_col].values.astype(float)
    y = y - y.mean()
    return t_hours, y


def compute_lombscargle(df, timestamp_col='dummy_timestamp', val_col='lfp_left_OvER_interpolate_z_scored',
                        min_period_hours=None, max_period_hours=48,
                        n_periods=4000, normalization='standard'):
    """
    Compute a Lomb-Scargle power spectrum from a timestamp/val dataframe.

    Parameters:
    - normalization: standard (power in [0,1], like R^2 of best-fit
      sinusoid; good for comparing across patients/recordings of
      different length or variance) or psd.
    - min_period_hours: floor for the period grid. Defaults to ~2x the
      median sampling interval (a rough analogue of the Nyquist limit).
    """
    t_hours, y = prep_series(df, timestamp_col, val_col)

    median_dt = np.median(np.diff(np.sort(t_hours)))
    if min_period_hours is None:
        min_period_hours = 2 * median_dt

    span_hours = t_hours[-1] - t_hours[0]
    print(f"n={len(t_hours)} samples, span={span_hours/24:.1f} days, "
          f"median dt={median_dt*60:.1f} min")
    print(f"Frequency resolution (~1/span) => can resolve periods "
          f"differing by roughly {1/span_hours*24**2:.2f}h near 24h period")

    periods_grid = np.linspace(min_period_hours,
                                min(max_period_hours, span_hours / 2),
                                n_periods)
    freqs_grid = 1 / periods_grid

    ls = LombScargle(t_hours, y, normalization=normalization)
    power = ls.power(freqs_grid)

    return ls, freqs_grid, periods_grid, power


def plot_lombscargle(periods, power, highlight=(6, 12, 24)):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(periods, power)
    for p in highlight:
        ax.axvline(p, color='r', linestyle='--', alpha=0.4)
    ax.set_xlabel('Period (hours)')
    ax.set_ylabel('Lomb-Scargle Power')
    ax.set_title('Power Spectrum')
    plt.tight_layout()
    return fig, ax


def extract_period_power(ls, target_periods=(6, 12, 24)):
    """
    Exact power at each target period -- no nearest-bin approximation,
    since Lomb-Scargle can be evaluated at any frequency directly.
    """
    target_freqs = 1 / np.array(target_periods, dtype=float)
    powers = ls.power(target_freqs)
    return dict(zip(target_periods, powers))


def normed_powers(periods, pxx):
    TARGET_H = (6, 12, 24)
    total = pxx.sum()
    return {p: pxx[np.abs(periods - p).argmin()] / total for p in TARGET_H}
