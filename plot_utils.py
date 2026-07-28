import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import matplotlib as mpl
import pandas as pd
import seaborn as sns # type: ignore
from datetime import timedelta, datetime, date, timezone
from statsmodels.stats.multitest import multipletests
import statsmodels.formula.api as smf
from statsmodels.tsa.stattools import acf
from scipy import stats
import re

epoch=date(1970, 1, 1)
epoch_dt=datetime.combine(epoch, datetime.min.time())

def transform_timestamp_to_days(pt_df, ax, tick_spacing_days, rotation=0, dbs_on_date=date(2030, 1, 1)):
    """
    Transforms the x-axis of a plot to represent days since DBS.

    Parameters:
    - pt_df (pd.DataFrame): DataFrame containing patient data.
    - ax (matplotlib.axes.Axes): The axes object to modify.
    - tick_spacing_days (int): The spacing between ticks in days.
    - rotation (float): Rotate the tick labels by this number of degrees.
    """

    xlim_orig = ax.get_xlim()

    dbs_on_since_epoch = (dbs_on_date - epoch).days

    min_time = pt_df['dummy_timestamp'].min()
    min_time_midnight = min_time.replace(hour=0, minute=0, second=0, microsecond=0)
    min_time_midnight_as_days_since = (min_time_midnight - epoch_dt) / timedelta(days=1)
    max_time = pt_df['dummy_timestamp'].max()
    max_time_midnight = max_time.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
    max_time_midnight_as_days_since = (max_time_midnight - epoch_dt) / timedelta(days=1)

    min_fractional_offset = min_time_midnight_as_days_since % 1
    max_fractional_offset = max_time_midnight_as_days_since % 1
    min_tick_adjusted = int(np.floor((min_time_midnight_as_days_since - dbs_on_since_epoch) / tick_spacing_days)) * tick_spacing_days + min_fractional_offset + dbs_on_since_epoch
    max_tick_adjusted = int(np.floor((max_time_midnight_as_days_since - dbs_on_since_epoch) / tick_spacing_days)) * tick_spacing_days + max_fractional_offset + dbs_on_since_epoch

    if max_tick_adjusted < max_time_midnight_as_days_since:
        max_tick_adjusted += tick_spacing_days
    
    ticks = np.arange(min_tick_adjusted, max_tick_adjusted + 1, tick_spacing_days)
    tick_labels = [int(np.round(tick - dbs_on_since_epoch)) for tick in ticks]
    # ax.set_xlim(min_tick_adjusted, max_tick_adjusted)
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels, rotation=rotation)
    ax.set_xlim(xlim_orig[0], xlim_orig[1])

def add_alpha_to_color(color, alpha):
    """
    Adds an alpha value to a given color.

    Parameters:
    - color: A color specified as a string (e.g., 'red', '#FF0000') or an RGB tuple (e.g., (1, 0, 0)).
    - alpha: A float between 0 and 1 representing the desired alpha value.

    Returns:
    - A color with the specified alpha value in RGBA format.
    """
    if isinstance(color, str):
        rgb = mpl.colors.to_rgb(color)
    elif isinstance(color, tuple) and len(color) == 3:
        rgb = color
    else:
        raise ValueError("Color must be a string or an RGB tuple.")
    
    rgba = (*rgb, alpha)
    return rgba

def compute_neff(days, vals, nlags=30):
    data_df = pd.DataFrame({'day': days, 'val': vals})
    full_days = np.arange(data_df['day'].min(), data_df['day'].max() + 1)
    data_df = (
        data_df.set_index('day')
        .reindex(full_days)
        .rename_axis('day')
        .reset_index()
    )
    x = data_df['val'].values
    mask = ~np.isnan(x)
    x_valid = x[mask]
    acf_vals = acf(x_valid, nlags=min(nlags, len(x_valid) - 1), fft=True)
    N = len(x_valid)
    rho = acf_vals[1:]
    pos_rho = rho[rho > 0]
    Neff = N / (1 + 2 * np.sum(pos_rho))
    return Neff

def welch_stats_with_effect_size(group1, group2, n1, n2):
    mean1, mean2 = np.mean(group1), np.mean(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    diff = mean1 - mean2

    # Welch's t-test (Satterthwaite dof) using n1/n2 as the sample sizes
    se = np.sqrt(var1 / n1 + var2 / n2)
    t_stat = diff / se
    dof = (var1 / n1 + var2 / n2) ** 2 / (
        (var1 / n1) ** 2 / (n1 - 1) + (var2 / n2) ** 2 / (n2 - 1)
    )
    p_val = 2 * stats.t.sf(np.abs(t_stat), df=dof)
    t_crit = stats.t.ppf(0.975, df=dof)
    ci = (diff - t_crit * se, diff + t_crit * se)

    df_pooled = n1 + n2 - 2
    pooled_sd = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / df_pooled)
    d = diff / pooled_sd
    J = 1 - 3 / (4 * df_pooled - 1)
    g = J * d
    var_d = (n1 + n2) / (n1 * n2) + d ** 2 / (2 * (n1 + n2))
    se_g = np.sqrt((J ** 2) * var_d)
    z_crit = stats.norm.ppf(0.975)
    g_ci = (g - z_crit * se_g, g + z_crit * se_g)

    return {
        't_stat': t_stat, 'p_val': p_val, 'dof': dof, 'ci': ci,
        'hedges_g': g, 'hedges_g_ci': g_ci, 'n1': n1, 'n2': n2,
    }

def fmt_ci(ci, decimals=3):
    lo, hi = ci
    if pd.isna(lo) or pd.isna(hi):
        return '-'
    return f'({lo:.{decimals}f}, {hi:.{decimals}f})'

def fmt_p(p):
    if pd.isna(p):
        return '-'
    return f'{p:.3e}'

def fmt_num(x, decimals=3):
    return '-' if pd.isna(x) else f'{x:.{decimals}f}'

def get_asterisk_str(p_val, alpha=0.05):
    if np.isnan(p_val):
        return r'$p$=NaN'
    elif p_val < 0.0001:
        return '****'
    elif p_val < 0.001:
        return '***'
    elif p_val < 0.01:
        return '**'
    elif p_val < alpha:
        return '*'
    else:
        return 'n.s.'

def interpolate_x_from_y(x_vals, y_vals, y_target):
    """
    Interpolates the x value corresponding to a given y value using linear interpolation.
    
    Parameters:
    - x_vals: Array of x values.
    - y_vals: Array of y values.
    - y_target: The y value for which to find the corresponding x value.

    Returns:
    - x_target: The interpolated x value corresponding to y_target.
    """
    x_vals = np.array(x_vals)
    y_vals = np.array(y_vals)

    # Eliminate indices where x_vals is equal to 0
    non_zero_indices = x_vals != 0
    x_vals = x_vals[non_zero_indices]
    y_vals = y_vals[non_zero_indices]

    # Ensure inputs are sorted by y (ascending)
    sort_idx = np.argsort(y_vals)
    x_vals = x_vals[sort_idx]
    y_vals = y_vals[sort_idx]

    # Find index where y_target would be inserted
    idx = np.searchsorted(y_vals, y_target)

    if idx == 0 or idx == len(y_vals):
        raise ValueError("y_target is out of bounds")

    return x_vals[idx]

def make_violin_plot_pretty(parts, color, median, ax, alpha=0.5, x_center=0):
    """
    Enhances the appearance of a violin plot by customizing its body color, transparency, 
    and adding a horizontal median line.

    Parameters:
    - parts: The components of the violin plot, obtained from the `ax.violinplot()` method.
    - color: The color to set for the violin body.
    - median: The y-coordinate of the median line to be drawn.
    - ax: The axes object where the violin plot is drawn.
    - alpha: The transparency level for the violin body (default is 0.5).
    """

    # Customize body
    body = parts['bodies'][0]
    body.set_facecolor(color)
    body.set_edgecolor(color)

    vertices = parts['bodies'][0].get_paths()[0].vertices
    ax.plot(vertices[:, 0], vertices[:, 1], color=color, lw=0.5, alpha=np.mean([alpha, alpha, 1.0]))
    x, y = vertices[:, 0], vertices[:, 1]

    x_target = interpolate_x_from_y(x, y, median)
    if np.isclose(x.min(), 0) or np.isclose(x.max(), 0):
        ax.hlines(median, xmin=x_target, xmax=x_center, color=color, lw=1, alpha=np.mean([alpha, alpha, 1.0]))
    else:
        ax.hlines(median, xmin=x_target, xmax=-x_target, color=color, lw=1, alpha=np.mean([alpha, alpha, 1.0]))

def plot_box_and_swarmplot(x, ys, ax, color=None, boxcolor=None, swarmcolor=None, widths=0.75,
                           alpha=0.5, edgecolor='#4c4c4c', lw=1, size=6):
    if color is None and boxcolor is None and swarmcolor is None:
        raise ValueError("At least one of color, boxcolor, or swarmcolor must be provided.")
    if color is not None and (boxcolor is not None or swarmcolor is not None):
        raise ValueError("If color is provided, boxcolor and swarmcolor should not be provided.")
    if (boxcolor is None and swarmcolor is not None) or (boxcolor is not None and swarmcolor is None):
        raise ValueError('Cannot specify boxcolor without swarmcolor, and vice versa. If you want to specify only one color, use the "color" argument instead.')
    if color is not None:
        boxcolor = color
        swarmcolor = color

    facecolor = add_alpha_to_color(boxcolor, alpha=alpha)
    ax.boxplot(
        ys, positions=[x], widths=widths,
        boxprops=dict(facecolor=facecolor, edgecolor=edgecolor, lw=lw),
        medianprops=dict(color=edgecolor, lw=lw),
        whiskerprops=dict(color=edgecolor, lw=lw),
        capprops=dict(color=edgecolor, lw=lw),
        showfliers=False, patch_artist=True, zorder=5
    )
    sns.swarmplot(
        x=[x] * len(ys),
        y=ys,
        color=swarmcolor,
        edgecolor=edgecolor,
        linewidth=1,
        ax=ax,
        size=size,
        zorder=7
    )
    return ax

def run_mixedlm(mlm_df, correction=True, verbose=False):
    model = smf.mixedlm(
        formula="power ~ response_status * time_bin",
        data=mlm_df,
        groups=mlm_df['pt_id']
    )
    result = model.fit(method="powell", reml=False)

    if verbose:
        print(f'MLM results:\n{result.summary()}')

    # Gather model results
    params = result.params
    pvalues = result.pvalues
    conf_int = result.conf_int()
    df_index = params.index.values
    groups = [t + ':time_bin[T.00:00:00]' if t == 'group[T.responder]' else t for t in df_index]

    # Create summary dataframe
    summary_df = pd.DataFrame({
        'coefficient': params,
        'pvalue': pvalues,
        'ci_lower': conf_int[0],
        'ci_upper': conf_int[1],
        'time_bins': groups
    })

    interaction_terms = summary_df[summary_df.index.str.contains('response_status[T.Responder]*')]

    # Apply multiple comparisons correction to p values
    if correction:
        _, interaction_terms['pvalue'], _, _ = multipletests(interaction_terms['pvalue'], method='fdr_bh')

    def clean_label(col):
        m = re.search(r'time_bin\[T\.(\d+):00-(\d+):00\]', col)
        return (int(m.group(1)), int(m.group(2))) if m else (0, 6)

    interaction_terms['time_bins'] = [clean_label(c) for c in interaction_terms['time_bins']]

    return summary_df, interaction_terms
