import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy import stats
from tqdm import tqdm

def fit_cosinor_linear(t, y, period_hours=24):
    """
    Fit a cosinor model to the data using least squares linear regression.

    Parameters:
    - t (array-like): Time values in hours (continuous across window).
    - y (array-like): Observed values.
    - period_hours (float): Period of the cosine function in hours.

    Returns:
    - M (float): Mean value of the fitted cosinor.
    - A (float): Amplitude of the fitted cosinor.
    - phi (float): Acrophase of the fitted cosinor in radians (0 to 2π).
    - t_peak_hours (float): Acrophase of the fitted cosinor in hours (0 to period_hours).
    """
    # t in hours (continuous across window); y is values
    omega = 2 * np.pi / period_hours
    X = np.column_stack([np.ones_like(t), np.cos(omega * t), np.sin(omega * t)])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)  # [M, beta, gamma]
    M, beta, gamma = coef
    A = np.sqrt(beta**2 + gamma**2)
    phi = np.arctan2(gamma, beta) % (2*np.pi)    # acrophase (radians, 0-2pi)
    t_peak_hours = (phi / omega) % period_hours  # acrophase (hours, 0-24)
    return M, A, phi, t_peak_hours, (beta, gamma)

def apply_sliding_cosinor(group_df, period_hours=24, window_size_days=3, hems=['left', 'right'],
                          lfp_col_suffix='_OvER_interpolate_z_scored', min_samples_per_window=144):
    """
    Apply a sliding window cosinor analysis to the data.

    Parameters:
    - group_df (pd.DataFrame): DataFrame containing the data for a single patient.
    - period_hours (float): Period of the cosine function in hours.
    - window_size_days (float): Size of the sliding window in days.
    - hems (list): List of hemispheres for which to perform the analysis.
    - lfp_col_suffix (str): Suffix for the LFP column names.
    - min_samples_per_window (int): Minimum number of samples required in a window to perform the fit.

    Returns:
    - result_df (pd.DataFrame): DataFrame containing the cosinor parameters for each window.
    """
    half_window_size = window_size_days // 2
    result = {'days_since_dbs': []}
    for hem in hems:
        result[f'{hem}_cosinor_acrophase_rad'] = []
        result[f'{hem}_cosinor_acrophase_hr'] = []
        result[f'{hem}_cosinor_amplitude'] = []
        result[f'{hem}_cosinor_mesor'] = []
        result[f'{hem}_cosinor_fit_cos_coef'] = []
        result[f'{hem}_cosinor_fit_sin_coef'] = []

    for day in np.arange(group_df['days_since_dbs'].min(), group_df['days_since_dbs'].max() + 1):
        window_df = group_df.loc[
            group_df['days_since_dbs'].between(day - half_window_size, day + half_window_size, inclusive='both')
        ]
        result['days_since_dbs'].append(day)

        for hem in hems:
            lfp_col = f'lfp_{hem}{lfp_col_suffix}'
            hem_window_df = window_df.dropna(subset=[lfp_col])

            fit_vals = (np.nan, np.nan, np.nan, np.nan, (np.nan, np.nan))
            if len(hem_window_df) >= min_samples_per_window:
                hours = hem_window_df['time_bin_time'].apply(lambda t: t.hour)
                mins = hem_window_df['time_bin_time'].apply(lambda t: t.minute)
                seconds = hem_window_df['time_bin_time'].apply(lambda t: t.second)
                t = (hours + mins / 60 + seconds / 3600)
                y = hem_window_df[lfp_col].values
                mask = ~np.isnan(y)
                if mask.any():
                    fit_vals = fit_cosinor_linear(t[mask], y[mask], period_hours=period_hours)

            M, A, phi, t_peak_hours, (beta, gamma) = fit_vals
            result[f'{hem}_cosinor_acrophase_rad'].append(phi)
            result[f'{hem}_cosinor_acrophase_hr'].append(t_peak_hours)
            result[f'{hem}_cosinor_amplitude'].append(A)
            result[f'{hem}_cosinor_mesor'].append(M)
            result[f'{hem}_cosinor_fit_cos_coef'].append(beta)
            result[f'{hem}_cosinor_fit_sin_coef'].append(gamma)

    return pd.DataFrame(result)

