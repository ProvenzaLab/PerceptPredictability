from scipy.signal import welch
from scipy import stats
import pandas as pd

WINDOW_DAYS = 4
RESAMPLE_RULE = '30min'
N_BINS = WINDOW_DAYS * 24 * 2
KEEP_STATES = ('Pre-DBS', 'Responder', 'Non-Responder')
TARGET_H = (6, 12, 24)
HEM_COL = 'lfp_left_OvER_interpolate_z_scored'

def _welch_periods(x):
    '''Welch PSD of a 30-min-sampled window, on an ascending period (hours) axis, DC dropped.'''
    f, pxx = welch(x, fs=1.0, nperseg=len(x))
    periods_h = 0.5 / f[1:]
    order = periods_h.argsort()
    return periods_h[order], pxx[1:][order]

def day_spectra(pt_df):
    '''Yield (state, periods_h, pxx) for every day that has a complete trailing window.'''
    series = pt_df.set_index('dummy_timestamp')[HEM_COL].resample(RESAMPLE_RULE).mean()
    states = pt_df.groupby(pt_df['dummy_timestamp'].dt.normalize())['state_label_str'].first()
    for day, state in states.items():
        if state not in KEEP_STATES:
            continue
        window = series.loc[day - pd.Timedelta(days=WINDOW_DAYS - 1):
                            day + pd.Timedelta(hours=23, minutes=30)].interpolate('linear')
        if len(window) != N_BINS or window.isna().any() or window.std() == 0:
            continue
        yield (state, *_welch_periods(stats.zscore(window.to_numpy())))

