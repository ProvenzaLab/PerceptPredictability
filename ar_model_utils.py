import numpy as np
import pandas as pd
from datetime import timedelta, datetime, date
import statsmodels.api as sm
from sklearn.metrics import r2_score


def predict_series_calc_R2(group: pd.DataFrame, ar_features: list, gt_colname: str, test_day: int,
                           new_cols: dict, window_size: int=3):
    """
    Fits autoregressive AR(1) model to data, then tests on a single day and returns predictions, daily R2 scores, prediction residuals, and residual variance.

    Parameters:
    - group (pd.DataFrame): A DataFrame containing processed data including state labels.
    - ar_features (list): A list of features to use for autoregression. If not None, this will be used instead of the num_lags parameter.
    - gt_colname (str): Name of the column for the feature the model is being trained to predict.
    - test_day (int): The day # (since DBS) to test the model on.
    - window_size (int): The size of the sliding window to use.

    Returns:
    - results_df (pd.DataFrame): A DataFrame containing predictions, daily R2 scores, prediction residuals, residual variance, and other spiking parameters.
    """

    all_days = group['days_since_dbs']
    unique_days = all_days.unique()
    
    test_df = group[all_days == test_day]
    train_df = group[all_days != test_day]

    results_df = pd.DataFrame(np.nan, index=test_df.index, columns=new_cols.values())
    
    if (unique_days[-1] - unique_days[0] != (window_size-1)) or (len(unique_days) != window_size): # Skip non-contiguous days
        return None
    
    train_df_no_na = train_df.dropna(subset=ar_features+[gt_colname])
    test_df_no_na = test_df.dropna(subset=ar_features+[gt_colname])
    if train_df_no_na.shape[0] < ((24 * 6) * (window_size - 2) + 1) or test_df_no_na.shape[0] < (24 * 6 // 2): # If we don't have enough data, just skip this day.
        return None

    model = sm.OLS(train_df_no_na[gt_colname], train_df_no_na[ar_features]).fit()
    phi_1 = model.params[ar_features[0]]

    # Generate predictions for the test data using the fitted model.
    preds = model.predict(test_df_no_na[ar_features])

    # Save predictions to dataframe
    results_df.loc[test_df_no_na.index, new_cols['preds']] = preds.values

    # Save phi_1 to dataframe
    results_df.loc[test_df_no_na.index, new_cols['phi_1']] = phi_1

    # Calculate all features from the predictions
    residuals = results_df[new_cols['preds']] - test_df[gt_colname]
    abs_residuals = np.abs(residuals)
    metrics_to_compute = {
        'residuals': lambda: residuals,
        'R2': lambda: r2_score(
            test_df_no_na[gt_colname],
            results_df.loc[test_df_no_na.index, new_cols['preds']]
        ),
    }

    for name, func in metrics_to_compute.items():
        if name in new_cols:
            results_df[new_cols[name]] = func()
    return results_df

def apply_sliding_window(g, ar_features: list, gt_colname: str, window_size: int=3):
    '''
    Apply a sliding window to a groupby object and return a DataFrame with the results of the sliding window.

    Parameters:
    - g (pd.DataFrame): A DataFrame containing processed data including state labels from a single lead.
    - ar_features (list): A list of features to use for autoregression. If not None, this will be used instead of the num_lags parameter.
    - gt_colname (str): Name of the column for the feature the model is being trained to predict.
    - window_size (int): The size of the sliding window to use.
    '''
    g = g.dropna(subset=gt_colname).copy()
    
    new_cols = {
        'phi_1': gt_colname.replace('z_scored', 'phi_1'),
        'preds': gt_colname.replace('z_scored', 'preds'),
        'residuals': gt_colname.replace('z_scored', 'residuals'),
        'R2': gt_colname.replace('z_scored', 'R2'),
    }

    if g.empty:
        return None
    unique_dates = g['days_since_dbs'].unique()
    results = []

    for this_date in unique_dates:
        td_arr = np.arange(-window_size+1, 1)
        date_list = td_arr + this_date
        day_mask = g['days_since_dbs'].isin(date_list)

        window = g[day_mask]
        day_results = predict_series_calc_R2(window, ar_features, gt_colname, this_date, new_cols, window_size)
        if day_results is not None:
            results.append(day_results)
    return pd.concat(results) if len(results) > 0 else None
