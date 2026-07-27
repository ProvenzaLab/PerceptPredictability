import numpy as np
import pandas as pd
from model_evaluation import delta_model

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

def generate_delta_avg_features(df, feature_cols, hem):
    df = delta_model(df, feature_cols)
    
    window_widths = [1, 3, 5, 7, 14, 21, 28]

    # container for all per-day results across pts and features
    per_day_results = []

    for feature in feature_cols:
        for mode in ['daily', 'delta']:
            if mode == 'delta':
                feature_col = f'delta_{feature}'
            else:
                feature_col = feature

            # single row per day per patient
            day_df = df.dropna(subset=[feature_col]).groupby(['pt_id', 'days_since_dbs'], as_index=False).first()
            if day_df.empty:
                continue

            results = []
            for pt_id, pt_day_df in day_df.groupby('pt_id', sort=False):
                days = pt_day_df['days_since_dbs'].to_numpy()
                values = pt_day_df[feature_col].to_numpy()
                rolling_matrix = group_calendar_rolling(days, values, window_widths)
                # build dataframe with pt_id and days_since_dbs so we can merge later
                cols = {f'{hem}{'_' if mode == 'daily' else '_delta_'}{feature.split('_')[-1]}_rolling_avg_{w}d': rolling_matrix[:, i]
                        for i, w in enumerate(window_widths)}
                res_df = pt_day_df[['pt_id', 'days_since_dbs']].copy()
                for cname, arr in cols.items():
                    res_df[cname] = arr
                results.append(res_df)

            if results:
                all_res = pd.concat(results, ignore_index=True)
                per_day_results.append(all_res)

    # if we computed anything, merge all per-day results into one dataframe and join back
    if per_day_results:
        per_day_df = pd.concat(per_day_results, ignore_index=True)
        # drop duplicates if same (pt_id, days_since_dbs, col) appear multiple times
        per_day_df = per_day_df.groupby(['pt_id', 'days_since_dbs']).first().reset_index()

        # merge: every row in df that has the same pt_id & days_since_dbs gets the per-day rolling values
        df = df.merge(per_day_df, on=['pt_id', 'days_since_dbs'], how='left', suffixes=('', '_perday'))

    return df

    