import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    confusion_matrix, roc_curve, roc_auc_score, balanced_accuracy_score, ConfusionMatrixDisplay, accuracy_score
)
from typing import Any, Dict, Tuple, Iterable
import plot_utils

COL_PAL = {
    "Pre-DBS": "#ffe900",
    "Disinhibited": "#ff0000",
    "Non-Responder": "#ffb900",
    "Responder": "#0000ff",
    "Transition": "#808080",
    "Unknown": "#808080"
}
default_colors = np.array(list(COL_PAL.values())).astype(object)

def leave_one_patient_out_logistic_regression(
    df: pd.DataFrame,
    feature_cols: Iterable,
    roc_fig: plt.Figure = None,
    roc_ax: plt.Axes = None,
    conf_mat_fig: plt.Figure = None,
    conf_mat_axs: plt.Axes = None,
    violin_fig: plt.Figure = None,
    violin_ax: plt.Axes = None,
    colors: list = None,
    shuffle: bool = False
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Perform leave-one-patient-out cross-validation with logistic regression.

    Args:
        df (pd.DataFrame): Dataframe containing data from one or more patients.
        feature_cols (list): Column names to be used as features.
        roc_fig (plt.Figure): Figure object for plotting ROC curve.
        roc_ax (plt.Axes): Axes object for plotting ROC curve.
        conf_mat_fig (plt.Figure): Figure object for plotting confusion matrices.
        conf_mat_axs (np.ndarray): Axes array for plotting confusion matrices.
        violin_fig (plt.Figure): Figure object for plotting violin plot of predictions.
        violin_ax (plt.Axes): Axes object for plotting violin plot of predictions.
        colors (list): List of colors for plotting.
        shuffle (bool): Whether to shuffle symptom state labels.

    Returns:
        Tuple[Dict[str, Any], Dict[str, Any]]: Overall results and patient-specific results.
    """
    if colors is None:
        colors = default_colors

    feature_cols = list(feature_cols)

    # Drop rows with missing feature values up front
    df = df.dropna(subset=feature_cols, how="any", ignore_index=True).copy()

    all_y_true, all_y_pred, all_y_prob, all_weights = [], [], [], []
    pt_results_dict = {}

    # Assign binary labels
    bad_labels = {"Unknown", "Transition", "Disinhibited"}
    label_map = {"Pre-DBS": 0, "Non-Responder": 0, "Responder": 1}
    df = df.loc[~df["state_label_str"].isin(bad_labels)].copy()
    df['label'] = df['state_label_str'].map(label_map)
    df = df.dropna(subset=['label']).copy()

    if shuffle:
        df['label'] = df['label'].sample(frac=1).values

    logo = LeaveOneGroupOut()
    groups = df["pt_id"].values

    for train_idx, test_idx in logo.split(df, groups=groups):
        train_df = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy()

        pt_id = test_df["pt_id"].iloc[0]
        pt_results: Dict[str, Any] = {}

        # Remove Pre-DBS data from test set if any delta features are used
        if any("delta" in col.lower() for col in feature_cols):
            test_df = test_df.query('state_label_str != "Pre-DBS"').copy()

        X_train = train_df[feature_cols]
        y_train = train_df["label"].astype(int)
        X_test = test_df[feature_cols]
        y_test = test_df["label"].astype(int)

        # Pipeline: standardize inside each fold, then fit logistic regression
        model = Pipeline(
            steps=[
                # ("scaler", StandardScaler()),
                ("logreg", LogisticRegression(
                    class_weight="balanced",
                    penalty=None,
                    solver="lbfgs",
                    max_iter=1000,),
                ),
            ]
        )
        if y_train.nunique() >= 2:
            model.fit(X_train, y_train)
        pt_results['model'] = model

        # Skip if there is no test data for this patient
        if test_df.shape[0] < 1:
            pt_results_dict[pt_id] = pt_results
            continue

        # Skip if training data has only one class
        if y_train.nunique() < 2:
            pt_results_dict[pt_id] = pt_results
            continue

        # Predict on fold
        y_pred = model.predict(X_test)
        y_prob_all = model.predict_proba(X_test)
        class_1_idx = np.where(model.named_steps["logreg"].classes_ == 1)[0][0]
        y_prob = y_prob_all[:, class_1_idx]

        # Store global results
        all_y_true.extend(y_test.to_numpy().tolist())
        all_y_pred.extend(y_pred.tolist())
        all_y_prob.extend(y_prob.tolist())
        all_weights.extend([1 / len(y_test)] * len(y_test))

        # Per-patient results
        pt_results["confusion_matrix"] = confusion_matrix(y_test, y_pred, labels=[0, 1])
        pt_results["weighted_confusion_matrix"] = confusion_matrix(
            y_test, y_pred, labels=[0, 1], normalize="all"
        )
        pt_results["y_true"] = y_test.to_numpy()
        pt_results["y_pred"] = y_pred
        pt_results["y_prob"] = y_prob
        pt_results["auc"] = roc_auc_score(y_test, y_prob) if y_test.nunique() > 1 else np.nan
        pt_results["balanced_accuracy"] = (
            balanced_accuracy_score(y_test, y_pred) if y_test.nunique() > 1 else np.nan
        )
        pt_results["raw_accuracy"] = accuracy_score(y_test, y_pred)

        pt_results_dict[pt_id] = pt_results

    all_y_true = np.asarray(all_y_true)
    all_y_pred = np.asarray(all_y_pred)
    all_y_prob = np.asarray(all_y_prob)
    all_weights = np.asarray(all_weights)

    overall_model = Pipeline(
        steps=[
            # ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(
                class_weight="balanced",
                penalty=None,
                solver="lbfgs",
                max_iter=1000,),
            ),
        ]
    )
    overall_model.fit(df[feature_cols], df["label"].astype(int))

    # Default empty outputs in case nothing was evaluated
    if all_y_true.size == 0:
        conf_matrix = np.zeros((2, 2), dtype=int)
        weighted_conf_matrix = np.zeros((2, 2), dtype=float)
        accuracy = np.nan
        weighted_accuracy = np.nan
        auc_score = np.nan
        weighted_auc_score = np.nan
        balanced_acc = np.nan
        weighted_balanced_acc = np.nan
        overall_tpr = overall_tnr = fpr = tpr = weighted_fpr = weighted_tpr = np.array([])
    else:
        conf_matrix = confusion_matrix(all_y_true, all_y_pred, labels=[0, 1])
        weighted_conf_matrix = confusion_matrix(
            all_y_true, all_y_pred, labels=[0, 1], sample_weight=all_weights
        )

        accuracy = accuracy_score(all_y_true, all_y_pred)
        weighted_accuracy = accuracy_score(all_y_true, all_y_pred, sample_weight=all_weights)

        auc_score = roc_auc_score(all_y_true, all_y_prob) if np.unique(all_y_true).size > 1 else np.nan
        weighted_auc_score = (
            roc_auc_score(all_y_true, all_y_prob, sample_weight=all_weights)
            if np.unique(all_y_true).size > 1
            else np.nan
        )

        balanced_acc = balanced_accuracy_score(all_y_true, all_y_pred)
        weighted_balanced_acc = balanced_accuracy_score(
            all_y_true, all_y_pred, sample_weight=all_weights
        )

        if np.unique(all_y_true).size > 1:
            fpr, tpr, _ = roc_curve(all_y_true, all_y_prob)
            weighted_fpr, weighted_tpr, _ = roc_curve(
                all_y_true, all_y_prob, sample_weight=all_weights
            )
        else:
            fpr = tpr = weighted_fpr = weighted_tpr = np.array([])

        tn, fp, fn, tp = conf_matrix.ravel()
        overall_tpr = tp / (tp + fn) if (tp + fn) > 0 else np.array([])
        overall_tnr = tn / (tn + fp) if (tn + fp) > 0 else np.array([])

    cols_string = "\n".join(feature_cols)

    # Plot ROC curves
    if roc_ax is not None:
        if fpr.size > 0:
            roc_ax.plot(fpr, tpr, label=f"Overall ROC (AUC = {auc_score:.4f})")
            roc_ax.plot(
                weighted_fpr,
                weighted_tpr,
                label=f"Weighted ROC (AUC = {weighted_auc_score:.4f})",
                linestyle="--",
            )
        roc_ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
        roc_ax.set(
            xlabel="False Positive Rate",
            ylabel="True Positive Rate",
            title=f"({cols_string})\nROC Curves for LOPO CV",
            xlim=[-0.01, 1.01],
            ylim=[-0.01, 1.01],
        )
        roc_ax.legend()

    # Plot confusion matrices
    if conf_mat_axs is not None:
        disp1 = ConfusionMatrixDisplay(
            confusion_matrix=conf_matrix,
            display_labels=["Non-Symptomatic", "Symptomatic"],
        )
        disp1.plot(ax=conf_mat_axs[0], cmap=plt.cm.viridis, values_format="d")
        conf_mat_axs[0].set_title(f"({cols_string})\nOverall Conf Mat for LOPO CV")

        disp2 = ConfusionMatrixDisplay(
            confusion_matrix=weighted_conf_matrix,
            display_labels=["Non-Symptomatic", "Symptomatic"],
        )
        disp2.plot(ax=conf_mat_axs[1], cmap=plt.cm.viridis, values_format=".2f")
        conf_mat_axs[1].set_title(f"({cols_string})\nWeighted Conf Mat for LOPO CV")
        conf_mat_fig.tight_layout()

    # Plot violin plot of feature distributions by true label
    if violin_ax is not None:

        if len(feature_cols) != 1:
            print(
                f"Warning: Feature columns should be a single column for violin plot. "
                f"Only using first feature column: {feature_cols[0]}"
            )
        violin_fig.tight_layout()

        feature_col = feature_cols[0]
        nr_data = df.query('state_label_str == "Pre-DBS" or state_label_str == "Non-Responder"')[feature_col]
        r_data = df.query('state_label_str == "Responder"')[feature_col]

        nr_parts = violin_ax.violinplot(nr_data, positions=[1], showextrema=False, side="low")
        r_parts = violin_ax.violinplot(r_data, positions=[1], showextrema=False, side="high")

        nr_median = np.median(nr_data)
        r_median = np.median(r_data)

        plot_utils.make_violin_plot_pretty(nr_parts, colors[0], nr_median, violin_ax, alpha=0.5, x_center=1)
        plot_utils.make_violin_plot_pretty(r_parts, colors[3], r_median, violin_ax, alpha=0.5, x_center=1)

        a = 0.1
        violin_ax.scatter(
            np.array([0.7] * len(nr_data)) + np.random.RandomState(42).normal(0, 0.01, size=len(nr_data)),
            nr_data,
            c=colors[0],
            marker="o",
            s=15,
            alpha=a,
        )
        violin_ax.scatter(
            np.array([1.3] * len(r_data)) + np.random.RandomState(42).normal(0, 0.01, size=len(r_data)),
            r_data,
            c=colors[3],
            marker="o",
            s=15,
            alpha=a,
        )

        violin_ax.set(
            xlim=[0.6, 1.4],
            xticks=[0.8, 1.2],
            xticklabels=["Symptomatic State", "Response State"],
            ylabel=f"{feature_cols}",
            title=(
                f"AUC Score: {auc_score:.4f}\n"
                f"Balanced Accuracy: {balanced_acc:.4f}\n"
                f"Weighted AUC Score: {weighted_auc_score:.4f}\n"
                f"Weighted Balanced Accuracy: {weighted_balanced_acc:.4f}"
            ),
        )
        violin_ax.tick_params("x", tick1On=False, tick2On=False)

    return {
        "confusion_matrix": conf_matrix,
        "weighted_confusion_matrix": weighted_conf_matrix,
        "raw_accuracy": accuracy,
        "weighted_raw_accuracy": weighted_accuracy,
        "AUC": auc_score,
        "weighted_AUC": weighted_auc_score,
        "balanced_accuracy": balanced_acc,
        "weighted_balanced_accuracy": weighted_balanced_acc,
        "true_positive_rate": overall_tpr,
        "true_negative_rate": overall_tnr
    }, pt_results_dict, overall_model


def plot_model_metrics(df, get_model_feature, window_widths,
                       boxplot_axs, roc_ax, conf_mat_axs,
                       operating_point=0.5, boxcolors=None, swarmcolors=None):
    """
    Plots model evaluation metrics (TPR, TNR, ROC curve, confusion matrix) for different window widths.

    Parameters:
    - df: DataFrame containing data for all patients.
    - get_model_feature: Function that takes a window width and returns the corresponding model feature column name.
    - window_widths: List of window widths to evaluate.
    - boxplot_axs: Tuple of two Matplotlib axes for plotting TPR and TNR boxplots.
    - roc_ax: Matplotlib axis for plotting the ROC curve.
    - conf_mat_axs: Tuple of two Matplotlib axes for plotting confusion matrices for specific window widths.
    - operating_point: Threshold for classifying probabilities into binary predictions (default is 0.5).
    - boxcolors: List of colors for the boxplots (default is None, which will use the default color).
    - swarmcolors: List of colors for the swarmplot points (default is ['#808080'] * len(window_widths)).

    Returns:
    - boxplot_axs: Updated axes with TPR and TNR boxplots.
    - roc_ax: Updated axis with ROC curve.
    - conf_mat_axs: Updated axes with confusion matrices.
    """
    if boxcolors is None:
        boxcolors = [None] * len(window_widths)
    if swarmcolors is None:
        swarmcolors = ['#808080'] * len(window_widths)
    for i, window_width in tqdm(enumerate(window_widths), total=len(window_widths)):
        model_feature = get_model_feature(window_width)
        model_df = df.dropna(subset=[model_feature], how='any').groupby(['pt_id', 'days_since_dbs']).head(1).reset_index(drop=True)
        results, pt_results, overall_model = leave_one_patient_out_logistic_regression(
            model_df,
            [model_feature],
            colors=list(COL_PAL.values()),
        )
        tprs, tnrs = [], []
        all_y_true, all_y_pred, all_y_prob = [], [], []
        for pt_id, pt_result in pt_results.items():
            if 'y_true' not in pt_result or 'y_prob' not in pt_result:
                continue
            y_true, y_prob = pt_result['y_true'], pt_result['y_prob']
            y_pred = (np.array(y_prob) >= operating_point).astype(int)
            conf_mat = confusion_matrix(y_true, y_pred, labels=[0, 1])
            tn, fp, fn, tp = conf_mat.ravel()
            tpr = tp / (tp + fn) if (tp + fn) > 0 else np.nan
            tnr = tn / (tn + fp) if (tn + fp) > 0 else np.nan
            tprs.append(tpr)
            tnrs.append(tnr)
            all_y_true.extend(y_true)
            all_y_pred.extend(y_pred)
            all_y_prob.extend(y_prob)
        tprs = [tp for tp in tprs if not np.isnan(tp)]
        tnrs = [tnr for tnr in tnrs if not np.isnan(tnr)]
        mean_tpr = np.mean(tprs)
        mean_tnr = np.mean(tnrs)
        all_y_true, all_y_pred, all_y_prob = np.array(all_y_true), np.array(all_y_pred), np.array(all_y_prob)

        plot_utils.plot_box_and_swarmplot(i, tprs, boxplot_axs[0], boxcolor=boxcolors[i],
                                          swarmcolor=swarmcolors[i], alpha=1, size=3)
        plot_utils.plot_box_and_swarmplot(i, tnrs, boxplot_axs[1], boxcolor=boxcolors[i],
                                          swarmcolor=swarmcolors[i], alpha=1, size=3)

        # get tnr and tpr of all_y_true and all_y_pred at the operating point
        boxplot_axs[0].scatter(i, mean_tpr, marker='^', color='g', s=50, zorder=5)
        boxplot_axs[1].scatter(i, mean_tnr, marker='^', color='g', s=50, zorder=5)

        if window_width in [1, 14]:
            # show roc curve
            roc_ax.plot([0, 1], [0, 1], linestyle='--', color='black')
            fpr, tpr, _ = roc_curve(all_y_true, all_y_prob)
            roc_auc = auc(fpr, tpr)
            color = boxcolors[i] if boxcolors[i] is not None else f'C{i}'
            roc_ax.plot(fpr, tpr, label=f'Window Width {window_width} days (AUC = {roc_auc:.2f})', color=color)

            # make a confusion matrix
            cm = confusion_matrix(all_y_true, all_y_pred, labels=[0, 1], normalize='true') * 100
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Non-Responder', 'Responder'])
            conf_mat_ax = conf_mat_axs[0 if window_width == 1 else 1]
            disp.plot(cmap=plt.cm.viridis, ax=conf_mat_ax, values_format='.2f', im_kw={'vmin': 0, 'vmax': 100})
            conf_mat_ax.set_title(f'Window Width {window_width} Days')
            for text in disp.text_.ravel():
                val = float(text.get_text())
                text.set_text(f'{val:.2f}%')

    roc_ax.legend()
    roc_ax.set(
        xlabel='False Positive Rate',
        ylabel='True Positive Rate',
        title='ROC Curve',
        xlim=[0, 1],
        ylim=[0, 1]
    )
    boxplot_axs[0].set(
        xticks=range(len(window_widths)), xticklabels=window_widths, xlabel='Window Width (days)',
        ylabel='True Positive Rate',title='TPR vs. Window Width', ylim=[0, 1]
    )
    boxplot_axs[1].set(
        xticks=range(len(window_widths)), xticklabels=window_widths, xlabel='Window Width (days)',
        ylabel='True Negative Rate', title='TNR vs. Window Width', ylim=[0, 1]
    )

    return boxplot_axs, roc_ax, conf_mat_axs

def delta_model(df, features):
    for id in df['pt_id'].unique():
        preDBS = df.query('pt_id == @id and days_since_dbs < 0')
        for f in features:
            df.loc[df['pt_id'] == id, f'delta_{f}'] = df[f] - np.nanmean(preDBS[f])
    return df
