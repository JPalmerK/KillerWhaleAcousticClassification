# -*- coding: utf-8 -*-
"""
Created on Sat Mar 15 23:29:54 2025

@author: kaity
"""



from keras.models import load_model
import EcotypeDefs as Eco
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import auc
import seaborn as sns
from sklearn.metrics import confusion_matrix
import statsmodels.api as sm

from sklearn.metrics import matthews_corrcoef

from sklearn.metrics import matthews_corrcoef

# === Add near your other helpers ===
def _recall_at_thresholds(eval_df, thresholds, truth_col="Truth"):
    """
    Compute per-class recall using class-specific thresholds.
    A row counts as TP for class c if Truth==c and score[c] >= thresholds[c].
    FN are Truth==c but score[c] < thresholds[c].
    Returns: dict {class_name: recall}
    """
    recalls = {}
    for c, thr in thresholds.items():
        is_truth = (eval_df[truth_col] == c)
        if is_truth.sum() == 0:
            recalls[c] = np.nan
            continue
        tp = ((is_truth) & (eval_df[c] >= thr)).sum()
        fn = ((is_truth) & (eval_df[c] < thr)).sum()
        recalls[c] = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    return recalls


def compute_mcc_for_srkw_tkw(cm_df):
    """
    Computes Matthews Correlation Coefficient (MCC) for binary classification 
    between SRKW and TKW using the provided confusion matrix.
    Assumes cm_df is indexed and columned by class name.
    """
    required_classes = {'SRKW', 'TKW'}
    present = required_classes.issubset(cm_df.index) and required_classes.issubset(cm_df.columns)
    if not present:
        print("MCC skipped: SRKW and TKW not both in confusion matrix.")
        return np.nan

    # Binary labels: SRKW = 1, TKW = 0
    TP = cm_df.at['SRKW', 'SRKW']
    TN = cm_df.at['TKW', 'TKW']
    FP = cm_df.at['SRKW', 'TKW']
    FN = cm_df.at['TKW', 'SRKW']

    numerator = TP * TN - FP * FN
    denominator = np.sqrt(
        (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)
    )

    if denominator == 0:
        return np.nan

    return numerator / denominator


def compute_mcc_for_srkw_hw(cm_df):
    """
    Computes Matthews Correlation Coefficient (MCC) for binary classification 
    between SRKW and TKW using the provided confusion matrix.
    Assumes cm_df is indexed and columned by class name.
    """
    required_classes = {'SRKW', 'TKW'}
    present = required_classes.issubset(cm_df.index) and required_classes.issubset(cm_df.columns)
    if not present:
        print("MCC skipped: SRKW and TKW not both in confusion matrix.")
        return np.nan

    # Binary labels: SRKW = 1, TKW = 0
    TP = cm_df.at['SRKW', 'SRKW']
    TN = cm_df.at['HW', 'HW']
    FP = cm_df.at['SRKW', 'HW']
    FN = cm_df.at['HW', 'SRKW']

    numerator = TP * TN - FP * FN
    denominator = np.sqrt(
        (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)
    )

    if denominator == 0:
        return np.nan

    return numerator / denominator
def plot_one_vs_others_roc(fp_df, all_df, relevant_classes=None, class_colors=None,
                           titleStr="One-vs-Others ROC Curve"):
    """
    Generates a One-vs-Others ROC curve for each class in the dataset.
    
    Parameters:
    - fp_df: DataFrame containing false positives (typically a subset of all_df)
    - all_df: Full dataset with 'Truth', 'Class', and 'Score' columns
    - relevant_classes: Optional list of class names to include in the comparison.
                        If None, all classes in the dataset will be used.
    - class_colors: Optional dictionary mapping class names to specific colors.
                    If None, uses Matplotlib's default 'tab10' colormap.
    - titleStr: Title for the ROC curve plot.

    Returns:
    - roc_data: Dictionary containing thresholds, false positive rate (FPR), and true positive rate (TPR) for each relevant class.
    """

    if relevant_classes is None:
        relevant_classes = np.unique(all_df['Class'])  # Use all classes if none are specified

    # Default colormap if no colors provided
    default_colors = {cls: plt.cm.get_cmap("tab10").colors[i % 10] for i, cls in enumerate(relevant_classes)}
    
    # Use provided colors or fall back to default
    color_map = default_colors if class_colors is None else {**default_colors, **class_colors}

    thresholds = np.linspace(0.35, 1, 100)  # Define the range of thresholds
    roc_data = {}  # Store ROC data for each relevant class

    plt.figure(figsize=(8, 6))

    for cls in relevant_classes:
        df_filtered = all_df[all_df['Truth'].isin(relevant_classes)].copy()
        df_class = fp_df[fp_df['Class'] == cls].copy()  # False positives subset
        tpData = df_filtered[df_filtered['Truth'] == cls].copy()  # True positives subset

        fpr, tpr = [], []

        for threshold in thresholds:
            # Apply threshold to determine predictions
            df_class['Predicted'] = df_class['Score'] >= threshold
            tpData['Predicted'] = (tpData['Score'] >= threshold) & (tpData['Class'] == cls)

            # Compute counts
            true_positive_count = tpData['Predicted'].sum()
            false_positive_count = df_class['Predicted'].sum()
            false_negative_count = len(tpData) - true_positive_count
            true_negative_count = len(df_filtered) - (true_positive_count + false_positive_count + false_negative_count)

            # Calculate FPR and TPR
            fpr_value = false_positive_count / (false_positive_count + true_negative_count) if (false_positive_count + true_negative_count) > 0 else 0
            tpr_value = true_positive_count / (true_positive_count + false_negative_count) if (true_positive_count + false_negative_count) > 0 else 0

            fpr.append(fpr_value)
            tpr.append(tpr_value)

        # Store results for this class
        roc_data[cls] = {'thresholds': thresholds, 'fpr': np.array(fpr), 'tpr': np.array(tpr)}

        # Plot the ROC curve
        plt.plot(fpr, tpr, label=cls, color=color_map.get(cls, "black"))  # Use class color or black fallback

    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title(titleStr)
    plt.legend()
    plt.grid()
    plt.show()

    return roc_data

def plot_confusion_matrix(
    data,
    threshold=0.5,
    class_column="Predicted Class",
    score_column="Score",
    truth_column="Truth",
    titleStr = 'Model x'):
    """
    Plots a confusion matrix (rows = predicted, columns = true) based on a 
    given threshold (float or dict), annotating the diagonal cells with recall.

    Parameters:
    - data: DataFrame containing:
        * class_column (str): The model's predicted label.
        * score_column (float): The model's confidence score.
        * truth_column (str): The actual (ground truth) label.
    - threshold: Either:
        * A float (e.g. 0.5): single global threshold, or
        * A dict mapping class_name (str) -> threshold (float).
          Classes not in the dict fall back to 0.5.
    - class_column: Column name for the predicted class (default "Class").
    - score_column: Column name for the confidence score (default "Score").
    - truth_column: Column name for the ground truth (default "Truth").

    Returns:
    - cm_df: The unnormalized confusion matrix DataFrame (with predicted classes
             as rows and true classes as columns). The plotted heatmap is 
             row-normalized and includes recall on the diagonal.
    """

    df = data.copy()

    # ------------------------------------------------
    # 1) Assign threshold per row (if dict) or global
    # ------------------------------------------------
    if isinstance(threshold, dict):
        def get_threshold_for_row(row):
            return threshold.get(row[class_column], 0.5)
        df["_ThresholdToUse"] = df.apply(get_threshold_for_row, axis=1)
    else:
        df["_ThresholdToUse"] = threshold

    # ------------------------------------------------
    # 2) Binarize predictions at the threshold
    # ------------------------------------------------
    df["Predicted"] = np.where(
        df[score_column] >= df["_ThresholdToUse"],
        df[class_column],
        "Background"
    )

    # ------------------------------------------------
    # 3) Compute confusion matrix in the usual (truth, predicted) order
    #    and then transpose it to flip axes.
    # ------------------------------------------------
    # By default, confusion_matrix has shape:
    #   (len(labels), len(labels)) => (row = truth, col = predicted).
    # We transpose it afterward so that row = predicted, col = truth.
    labels = df[class_column].unique()  # or union of all truth/predict
    cm_normal = confusion_matrix(
        df[truth_column],
        df["Predicted"],
        labels=labels
    )

    # Flip (row=>predicted, col=>truth) by transposing
    cm_flipped = cm_normal.T

    # Create a DataFrame for unnormalized confusion matrix
    # index = predicted classes, columns = true classes
    cm_df = pd.DataFrame(
        cm_flipped,
        index=labels,
        columns=labels
    )

    # ------------------------------------------------
    # 4) Row-normalize this flipped matrix for display
    #    (so each row sums to 1).
    # ------------------------------------------------
    cm_df_norm = cm_df.div(cm_df.sum(axis=0), axis=1)

    # Remove empty rows (if any) from both unnormalized & normalized
    cm_df.dropna(axis=1, how='all', inplace=True)
    cm_df_norm.dropna(axis=1, how='all', inplace=True)

    # ------------------------------------------------
    # 5) Calculate recall for each class:
    #    Recall = TP / (total times class appears in ground truth).
    # ------------------------------------------------
    # Diagonal = predicted == c, truth == c. 
    # For recall, we need how many times c truly appears in `truth_column`.
    RecallArray = pd.DataFrame()
    RecallArray["Class"] = cm_df.index  # predicted classes

    # (A) True Positives on the diagonal (in flipped matrix)
    RecallArray["TP"] = np.diag(cm_df.values)

    # (B) Count how many times each class is in the truth
    truth_counts = df[truth_column].value_counts()

    # Map each predicted class to how often it appears in truth
    # (If a predicted class never appears as truth, recall is NaN)
    RecallArray["TruthCount"] = RecallArray["Class"].map(truth_counts).fillna(0).astype(int)
    RecallArray["Recall"] = RecallArray["TP"] / RecallArray["TruthCount"]
    RecallArray.loc[RecallArray["TruthCount"] == 0, "Recall"] = np.nan

    # ------------------------------------------------
    # 6) Plot the flipped & row-normalized matrix
    #    With recall on the diagonal cells
    # ------------------------------------------------
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(
        cm_df_norm,
        annot=False,  # We'll place text manually
        cmap="Blues",
        linewidths=0.5
    )

    n_rows, n_cols = cm_df_norm.shape

    for i in range(n_rows):
        for j in range(n_cols):
            val = cm_df_norm.iloc[i, j]
            x_coord = j + 0.5
            y_coord = i + 0.5

            # Determine text color based on background brightness
            text_color = "white" if val > 0.5 else "black"

            if i == j:
                cls_name = cm_df_norm.index[i]
                row_info = RecallArray[RecallArray["Class"] == cls_name]
                if not row_info.empty:
                    rec_val = row_info["Recall"].values[0]
                    if pd.notnull(rec_val):
                        text_str = f"{val:.2f}\nRecall: {rec_val:.2f}"
                    else:
                        text_str = f"{val:.2f}\nRecall: N/A"
                else:
                    text_str = f"{val:.2f}"
            else:
                text_str = f"{val:.2f}"

            ax.text(
                x_coord,
                y_coord,
                text_str,
                ha="center",
                va="center",
                color=text_color,
                fontsize=12
            )


    plt.ylabel("Predicted Class")
    plt.xlabel("True Class")
    plt.title(titleStr)
    plt.show()

    # Clean up
    df.drop(columns=["_ThresholdToUse"], inplace=True, errors="ignore")
    return cm_df

def plot_confusion_matrix_multilabel(
    data,
    score_cols,
    truth_column="Truth",
    threshold=0.5,
    titleStr = 'Model x'):
    """
    Plots a confusion matrix for a multi-label classifier that outputs scores
    for multiple classes in each row. One row can surpass thresholds for more
    than one class, thus predicting multiple classes simultaneously.

    This version:
      - Removes 'Background' rows/columns from the final confusion matrix,
      - Removes any rows that are all NaN after normalization,
      - Forces the row and column order to match the order in `score_cols`.

    Parameters:
    - data: A pandas DataFrame with:
        * One column for the ground truth (truth_column).
        * Multiple columns (score_cols) for each class's prediction score.
    - score_cols: List of columns corresponding to each possible class's score,
                  e.g. ["SRKW", "TKW", "OKW", "HW"].
    - truth_column: Name of the ground-truth column in `data`.
    - threshold: Either:
        * A float (applies the same threshold to all classes), OR
        * A dict mapping class_name -> float threshold (others default to 0.5).

    Returns:
    - cm_df: The unnormalized multi-label confusion matrix (rows = true classes,
      columns = predicted classes), with 'Background' removed, row/column
      order forced to match `score_cols`, and any all-NaN rows dropped from
      the normalized heatmap.
    """

    # ---------------------------
    # 1) Determine the classes & thresholds
    # ---------------------------
    # We'll add "Background" for predictions if no class surpasses threshold.
    classes = list(score_cols)            # e.g. ["SRKW", "TKW", "OKW", "HW"]
    classes_with_bg = classes + ["Background"]

    def get_threshold_for_class(c):
        if isinstance(threshold, dict):
            return threshold.get(c, 0.5)  # default to 0.5 if missing
        else:
            return threshold  # single float

    # ---------------------------
    # 2) Build predicted sets for each row
    # ---------------------------
    predicted_sets = []
    for _, row in data.iterrows():
        row_predicted = []
        for c in classes:
            score_val = row[c]
            thr_val = get_threshold_for_class(c)
            if pd.notnull(score_val) and score_val >= thr_val:
                row_predicted.append(c)
        if not row_predicted:
            row_predicted = ["Background"]
        predicted_sets.append(row_predicted)

    # ---------------------------
    # 3) Identify possible truth labels
    # ---------------------------
    all_true_labels = sorted(data[truth_column].unique().tolist())
    # If ground truth can be "Background", handle it, otherwise add it
    if "Background" not in all_true_labels:
        all_possible_truth = all_true_labels + ["Background"]
    else:
        all_possible_truth = all_true_labels

    # Prepare array [num_truth_labels x num_pred_labels]
    cm_array = np.zeros((len(all_possible_truth), len(classes_with_bg)), dtype=int)

    # ---------------------------
    # 4) Fill the unnormalized confusion matrix
    # ---------------------------
    for i, row in enumerate(data.itertuples(index=False)):
        true_label = getattr(row, truth_column)
        if true_label not in all_possible_truth:
            true_label = "Background"
        row_idx = all_possible_truth.index(true_label)

        preds = predicted_sets[i]
        for p in preds:
            if p not in classes_with_bg:
                p = "Background"
            col_idx = classes_with_bg.index(p)
            cm_array[row_idx, col_idx] += 1

    # Convert to DataFrame
    cm_df = pd.DataFrame(cm_array, index=all_possible_truth, columns=classes_with_bg)

    # ---------------------------
    # 5) Remove "Background"
    # ---------------------------
    if "Background" in cm_df.index:
        cm_df.drop("Background", axis=0, inplace=True)
    if "Background" in cm_df.columns:
        cm_df.drop("Background", axis=1, inplace=True)

    # ---------------------------
    # 6) Force row/column order to match `score_cols`
    # ---------------------------
    # We'll only keep the classes that actually appear in `score_cols`,
    # but forcibly reindex in exactly that order. Missing classes get zero counts.
    cm_df = cm_df.reindex(index=score_cols, columns=score_cols, fill_value=0)

    # ---------------------------
    # 7) Create a row-normalized version and drop rows that become all NaN
    # ---------------------------
    cm_df_norm = cm_df.div(cm_df.sum(axis=1), axis=0)
    cm_df_norm.dropna(axis=0, how='all', inplace=True)

    # ---------------------------
    # 8) Compute recall: TP / total times class appears in ground truth
    # ---------------------------
    truth_counts = data[truth_column].value_counts()
    recall_vals = {}
    for cls_name in cm_df.index:
        if cls_name not in cm_df.columns:
            tp = 0
        else:
            tp = cm_df.loc[cls_name, cls_name]
        tot_truth = truth_counts.get(cls_name, 0)
        recall_vals[cls_name] = tp / tot_truth if tot_truth > 0 else np.nan

    # ---------------------------
    # 9) Plot the row-normalized matrix
    # ---------------------------
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(
        cm_df_norm,
        annot=False,
        cmap="Blues",
        linewidths=0.5
    )

    # Manually annotate each cell
    n_rows = cm_df_norm.shape[0]
    n_cols = cm_df_norm.shape[1]

    for i in range(n_rows):
        for j in range(n_cols):
            val = cm_df_norm.iloc[i, j]
            x_coord = j + 0.5
            y_coord = i + 0.5

            if i == j:
                # Diagonal => add recall
                cls_name = cm_df_norm.index[i]
                rec_val = recall_vals.get(cls_name, np.nan)
                if pd.notnull(rec_val):
                    text_str = f"{val:.2f}\nRecall: {rec_val:.2f}"
                else:
                    text_str = f"{val:.2f}\nRecall: N/A"
                text_color = "white"
            else:
                text_str = f"{val:.2f}"
                text_color = "black"

            ax.text(
                x_coord,
                y_coord,
                text_str,
                ha="center",
                va="center",
                color=text_color,
                fontsize=12
            )

    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("True Class")
    ax.set_title(titleStr)
    plt.show()

    return cm_df

def plot_logistic_fit_with_cutoffs(data, score_column="Score", 
                                   class_column="Predicted Class", truth_column="Truth",
                                   titleStr = ":SRKW", plotLogit= False):
    """
    Fits logistic regression models to estimate the probability of correct detection 
    as a function of BirdNET confidence score, following Wood & Kahl (2024).
    
    Plots the logistic curve and thresholds for 90%, 95%, and 99% probability.
    
    Parameters:
    - data: DataFrame containing BirdNET evaluation results.
    - score_column: Column with BirdNET confidence scores (0-1).
    - class_column: Column with the predicted class.
    - truth_column: Column with the true class.

    Returns:
    - Logistic regression results for confidence and logit-transformed scores.
    """

    # Copy data to avoid modifying original
    data = data.copy()

    # Create binary column for correct detection
    data["Correct"] = (data[class_column] == data[truth_column]).astype(int)

    # Logit transformation of score (avoiding log(0) by clipping values)
    eps = 1e-9  # Small value to prevent log(0) errors
    data["Logit_Score"] = np.log(np.clip(data[score_column], eps, 1 - eps) / np.clip(1 - data[score_column], eps, 1 - eps))

    # Fit logistic regression models
    conf_model = sm.Logit(data["Correct"],  
                          sm.add_constant(data[score_column])).fit(disp=False)
    logit_model = sm.Logit(data["Correct"], 
                           sm.add_constant(data["Logit_Score"])).fit(disp=False)

    # Generate prediction ranges
    conf_range = np.linspace(0.01, 0.99, 1000)  # Confidence score range
    logit_range = np.linspace(data["Logit_Score"].min(), data["Logit_Score"].max(), 1000)

    # Predict probabilities
    conf_pred = conf_model.predict(sm.add_constant(conf_range))
    logit_pred = logit_model.predict(sm.add_constant(logit_range))

    # Compute score cutoffs for 90%, 95%, and 99% probability thresholds
    def find_cutoff(model, coef_index):
        return (np.log([0.90 / 0.10, 0.95 / 0.05, 0.99 / 0.01]) - model.params[0]) / model.params[coef_index]

    conf_cutoffs = find_cutoff(conf_model, 1)
    logit_cutoffs = find_cutoff(logit_model, 1)
    
    if plotLogit:

        # Plot Confidence Score Model
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        sns.scatterplot(x=data[score_column], y=data["Correct"], alpha=0.3, label="Observations")
        plt.plot(conf_range, conf_pred, color="red", label="Logistic Fit (Confidence Score)")
        for i, cutoff in enumerate(conf_cutoffs):
            plt.axvline(cutoff, linestyle="--", color=["orange", "red", "magenta"][i], label=f"p={0.9 + i*0.05:.2f}")
        plt.xlabel("BirdNET Confidence Score")
        plt.ylabel("Pr(Correct Detection)")
        plt.title(f"Logistic Fit: Confidence Score {titleStr}")
        plt.legend()
        plt.grid()
    
        # Plot Logit Score Model
        plt.subplot(1, 2, 2)
        sns.scatterplot(x=data["Logit_Score"], y=data["Correct"], alpha=0.3, label="Observations")
        plt.plot(logit_range, logit_pred, color="blue", label="Logistic Fit (Logit Score)")
        for i, cutoff in enumerate(logit_cutoffs):
            plt.axvline(cutoff, linestyle="--", color=["orange", "red", "magenta"][i], label=f"p={0.9 + i*0.05:.2f}")
        plt.xlabel(f"Logit of BirdNET Confidence Score {titleStr}")
        plt.ylabel("Pr(Correct Detection)")
        plt.title("Logistic Fit: Logit Score")
        plt.legend()
        plt.grid()
    
        plt.show()

    return conf_model, logit_model

def plot_one_vs_others_pr(all_df, 
                          relevant_classes=None, 
                          class_colors=None,
                          titleStr="One-vs-Others Precision–Recall Curve"):
    """
    Generates a One-vs.-Others Precision–Recall curve for each class in the dataset,
    deriving false positives and true positives internally.

    Parameters:
    - all_df: DataFrame containing 'Truth', 'Class', and 'Score' columns.
    - relevant_classes: List of class names to include in the comparison. If None, uses all classes.
    - class_colors: Dictionary mapping class names to specific colors.
    - titleStr: Title for the Precision–Recall curve plot.

    Returns:
    - pr_data: Dictionary containing precision-recall data for each class.
    - auc_pr_dict: Dictionary containing AUC-PR values for each class.
    - mean_ap: Mean Average Precision (mAP) across all classes.
    """

    # If no classes specified, use all unique ones from 'Class'
    if relevant_classes is None:
        relevant_classes = np.unique(all_df['Class'])
        
    truthClasses = np.unique(all_df['Truth'])

    # Assign a color to each class if not provided
    default_colors = {
        cls: plt.cm.get_cmap("tab10").colors[i % 10]
        for i, cls in enumerate(relevant_classes)
    }
    color_map = default_colors if class_colors is None else {**default_colors, **class_colors}

    # Define thresholds to sweep over
    thresholds = np.linspace(0, 1, 200)
    #thresholds = np.unique(all_df['Top Score'])

    # Storage for precision–recall data and AUC-PR values
    pr_data = {}
    auc_pr_dict = {}  # Separate dictionary for AUC-PR values

    plt.figure(figsize=(8, 6))

    for cls in relevant_classes:
        precision_list, recall_list = [], []
    
        # Binary masks for positive class (truth)
        is_truth_cls = all_df['Truth'] == cls

        for threshold in thresholds:
            # TP: Correct predictions above threshold
            TP = ((is_truth_cls) & (all_df[cls] >= threshold)).sum()
    
            # FP: Incorrect predictions above threshold
            FP = ((~is_truth_cls) & (all_df[cls] >= threshold)).sum()
    
            # FN: Missed positives (either wrong class or too low score)
            FN = (is_truth_cls & (all_df[cls] < threshold)).sum()
    
            # Precision calculation
            precision = TP / (TP + FP) if (TP + FP) > 0 else 1.0
    
            # Recall calculation
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    
            precision_list.append(precision)
            recall_list.append(recall)
        
        # Compute area under PR curve (AUC-PR)
        auc_pr = auc(recall_list, precision_list)

        # Store results
        pr_data[cls] = {
            'thresholds': thresholds,
            'precision': np.array(precision_list),
            'recall': np.array(recall_list)
        }

        # Store AUC-PR in a separate dictionary
        auc_pr_dict[cls] = auc_pr

        # Plot Precision–Recall curve
        plt.plot(recall_list, precision_list, label=cls, 
                 color=color_map.get(cls, "black"))

    # Compute mean average precision (mAP)
    mean_ap = np.mean(list(auc_pr_dict.values()))

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(titleStr)
    plt.legend()
    plt.grid()
    plt.show()

    return pr_data, auc_pr_dict, mean_ap  # Return mAP as a separate output

def plot_allKW_PR(all_df, 
                          KW_truthNames=['SRKW', 'TKW', 'OKW', 'KW_und', 'NRKW'], 
                          class_colors=None,
                          titleStr="One-vs-Others Precision–Recall Curve"):
    """
    Generates a Precision–Recall curve for all killer whale calls. Anything
    KW annotation, regardless of ecotype or not, that is identified as any 
    one of the KW Ecotypes in the model output is considered a true positive

    Parameters:
    - all_df: DataFrame containing 'Truth', 'Class', and 'Score' columns.
    - relevant_classes: List of class names to include in the comparison. If None, uses all classes.
    - class_colors: Dictionary mapping class names to specific colors.
    - titleStr: Title for the Precision–Recall curve plot.

    Returns:
    - pr_data: Dictionary containing precision-recall data for each class.
    - auc_pr_dict: Dictionary containing AUC-PR values for each class.
    - mean_ap: Mean Average Precision (mAP) across all classes.
    """
    
    
    all_df['isKW'] = all_df['Truth'] in KW_truthNames
    
    
    # Define thresholds to sweep over
    thresholds = np.linspace(0, 1, 200)


    # Storage for precision–recall data and AUC-PR values
    pr_data = {}
    auc_pr_dict = {}  # Separate dictionary for AUC-PR values

    plt.figure(figsize=(8, 6))

    precision_list, recall_list = [], []

    # Binary masks for positive class (truth)
    is_truth_cls = all_df['Truth'] in KW_truthNames

    for threshold in thresholds:
        # TP: Correct predictions above threshold
        TP = ((is_truth_cls) & (all_df['Top Score'] >= threshold)).sum()

        # FP: Incorrect predictions above threshold
        FP = ((~is_truth_cls) & (all_df['Top Score'] >= threshold)).sum()

        # FN: Missed positives (either wrong class or too low score)
        FN = (is_truth_cls & (all_df['Top Score'] < threshold)).sum()

        # Precision calculation
        precision = TP / (TP + FP) if (TP + FP) > 0 else 1.0

        # Recall calculation
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0

        precision_list.append(precision)
        recall_list.append(recall)
    
    # Compute area under PR curve (AUC-PR)
    auc_pr = auc(recall_list, precision_list)

    # Store results
    pr_data = {
        'thresholds': thresholds,
        'precision': np.array(precision_list),
        'recall': np.array(recall_list)
    }

    # Store AUC-PR in a separate dictionary
    auc_pr_dict = auc_pr

    # Plot Precision–Recall curve
    plt.plot(recall_list, precision_list, color= "black")

    # Compute mean average precision (mAP)
    mean_ap = np.mean(list(auc_pr_dict.values()))

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(titleStr)
    plt.legend()
    plt.grid()
    plt.show()

    return pr_data, auc_pr_dict, mean_ap  # Return mAP as a separate output


import matplotlib.pyplot as plt

def compare_pr_curves(metrics_list, model_labels, target_class, title=None):
    """
    Plot precision-recall curves for a target class across multiple models.

    Parameters:
    - metrics_list: List of metric dictionaries (each structured like metrics_DCLDE_01)
    - model_labels: List of names for each model (used in plot legend)
    - target_class: String, e.g. 'SRKW', 'TKW', 'HW'
    - title: Optional title for the plot
    """
    if len(metrics_list) != len(model_labels):
        raise ValueError("metrics_list and model_labels must be the same length")

    plt.figure(figsize=(8, 6))

    for metrics, label in zip(metrics_list, model_labels):
        try:
            pr = metrics['pr_data'][target_class]
            plt.plot(pr['recall'], pr['precision'], label=f"{label} (AUC: {metrics['AUC'].get(target_class, 'N/A'):.3f})")
        except KeyError:
            print(f"Warning: Class '{target_class}' not found in model '{label}'")
            continue

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title or f'Precision–Recall Curve for Class: {target_class}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def find_cutoff(model, coef_index):
    return (np.log([0.90 / 0.10, 0.95 / 0.05, 0.99 / 0.01]) - model.params[0]) / model.params[coef_index]

# --- Helper Functions ---

def process_dataset(model_path, label_path, audio_folder, truth_label,
                    output_csv="predictions_output.csv", 
                    return_scores=True, 
                    classes=['SRKW','OKW','HW','TKW'],
                    thresh=0):
    """
    Process an audio folder using Eco.BirdNetPredictor.
    Returns a DataFrame with raw scores, the predicted class (as the max among classes),
    and a 'Truth' column set to the given truth_label.
    """
    processor = Eco.BirdNetPredictorNew(model_path, label_path, 
                                        audio_folder, 
                                        confidence_thresh=thresh)
    df = processor.batch_process_audio_folder(output_csv, 
                                              return_raw_scores=True)
    return df
    #df['Class_Est'] = df[classes].idxmax(axis=1)
    
    
    # if return_scores:
    #     df, scores = processor.batch_process_audio_folder(output_csv, 
    #                                                       return_raw_scores=True)
    #     scores['Class_Est'] = scores[classes].idxmax(axis=1)
    #     scores['Truth'] = truth_label
    #     return scores
    # else:
    #     return processor.batch_process_audio_folder(output_csv)

def process_multiple_datasets(model_path, label_path, folder_truth_map,
                              output_csv="predictions_output.csv", 
                              classes=['SRKW','OKW','HW','TKW'], 
                              thresh = 0):
    """
    Process multiple datasets defined by a dictionary mapping truth labels to folder paths.
    Returns a combined DataFrame.
    
    allPreds: logical, whether to threshold 
    """
    datasets = []
    for truth, folder in folder_truth_map.items():
        df = process_dataset(model_path, label_path, folder, truth,
                             output_csv=output_csv, classes=classes,
                             thresh=thresh)
        datasets.append(df)
    return pd.concat(datasets, ignore_index=True)

def compute_threshold(data, score_column, class_column, plotLogit = False, 
                      title_suffix=""):
    """
    Compute a threshold cutoff for a given class using a logistic fit.
    (Assumes plot_logistic_fit_with_cutoffs and find_cutoff are available.)
    
    Returns the 90th percentile
    """
    logit, _ = plot_logistic_fit_with_cutoffs(data, 
                                              score_column=score_column, 
                                              class_column=class_column,
                                              truth_column="Truth", 
                                              titleStr=title_suffix,
                                              plotLogit=plotLogit)
    cutoff = find_cutoff(logit, 1)[0]
    return cutoff

def evaluate_model(eval_df, 
                   custom_thresholds, 
                   pr_title="Precision–Recall Curve", 
                   roc_title="ROC Curve", 
                   plotPR = True, 
                   plotROC= False):
    """
    Given an evaluation DataFrame and custom thresholds, compute scores,
    identify false positives, and then plot PR curve, ROC curve, and confusion matrix.
    Returns the confusion matrix DataFrame.
    """
    # Compute score for each row based on its predicted class
    eval_df['Score'] = eval_df.apply(lambda row: row[row['Predicted Class']], axis=1)
    
    
    
    # Mark false positives
    eval_df['FP'] = eval_df['Predicted Class'] != eval_df['Truth']
    fp_data = eval_df[eval_df['FP']]
    

    
    metrics = dict({})
    # Plot ROC curve (using preset colors)
    class_colors = {'SRKW': '#1f77b4', 'TKW': '#ff7f0e', 'HW': '#2ca02c', 'OKW': '#e377c2'}
    if plotPR:
        # Plot Precision–Recall curve
        pr_data, auc_pr_dict, mean_ap = plot_one_vs_others_pr_with_kwund(
            eval_df, relevant_classes=list(custom_thresholds.keys()), 
                              class_colors=None, titleStr=pr_title)
       
        metrics['AUC'] = auc_pr_dict
        metrics['pr_data']= pr_data
        metrics['MAP'] = mean_ap
    

    if plotROC:
        plot_one_vs_others_roc(fp_data, eval_df,
                               titleStr=roc_title, 
                               class_colors=class_colors)
    
    # Plot and return confusion matrix
    cm_df = plot_confusion_matrix(eval_df,
                                  threshold=custom_thresholds,
                                    titleStr =roc_title)
    metrics['cm']= cm_df
    metrics['mcc_srkw_tkw'] = compute_mcc_for_srkw_tkw(cm_df)
    metrics['mcc_srkw_hw'] = compute_mcc_for_srkw_hw(cm_df)
    
    # --- NEW: recall at your logistic thresholds (the P90s you computed) ---
    metrics['recall_at_p90'] = _recall_at_thresholds(eval_df, custom_thresholds)


    
    return metrics


def plot_one_vs_others_pr_with_kwund(all_df, 
                                     relevant_classes=None, 
                                     class_colors=None,
                                     titleStr="One-vs-Others Precision–Recall Curve (w/ KW_und)"):
    """
    Plots one-vs-all precision-recall curves for each known class, excluding KW_und.
    Then separately evaluates a PR curve for KW_und annotations where any known KW prediction
    is treated as a true positive, and any background prediction is a false positive.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import auc

    # Define full sets
    all_known_kw_labels = ['SRKW', 'TKW', 'OKW', 'NRKW']
    # Only keep those that are actually in the DataFrame
    present_kw_labels = [label for label in all_known_kw_labels if label in all_df.columns]

    if not present_kw_labels:
        raise ValueError("None of the known KW label columns (SRKW, TKW, OKW, NRKW) were found in the DataFrame.")

    # Define background columns dynamically: any column not a KW label or meta
    meta_cols = ['Truth', 'Class', 'Score']
    background_labels = [c for c in all_df.columns 
                         if c not in present_kw_labels + ['KW_und'] + meta_cols]

    # Separate known ecotype examples from KW_und
    df_known = all_df[all_df['Truth'] != 'KW_und'].copy()
    df_kwund = all_df[all_df['Truth'] == 'KW_und'].copy()

    if relevant_classes is None:
        relevant_classes = np.unique(df_known['Class'])

    default_colors = {
        cls: plt.colormaps["tab10"].colors[i % 10]
        for i, cls in enumerate(relevant_classes)
    }
    color_map = default_colors if class_colors is None else {**default_colors, **class_colors}

    thresholds = np.linspace(0, 1, 200)
    pr_data = {}
    auc_pr_dict = {}

    plt.figure(figsize=(8, 6))

    # ---- PR curves for known classes ----
    for cls in relevant_classes:
        precision_list, recall_list = [], []
        is_truth_cls = df_known['Truth'] == cls

        for threshold in thresholds:
            TP = ((is_truth_cls) & (df_known[cls] >= threshold)).sum()
            FP = ((~is_truth_cls) & (df_known[cls] >= threshold)).sum()
            FN = ((is_truth_cls) & (df_known[cls] < threshold)).sum()

            precision = TP / (TP + FP) if (TP + FP) > 0 else 1.0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0

            precision_list.append(precision)
            recall_list.append(recall)

        auc_pr = auc(recall_list, precision_list)

        pr_data[cls] = {
            'thresholds': thresholds,
            'precision': np.array(precision_list),
            'recall': np.array(recall_list)
        }
        auc_pr_dict[cls] = auc_pr

        plt.plot(recall_list, precision_list, label=cls, 
                 color=color_map.get(cls, "black"))

    # ---- Special case: KW_und as TP if any KW label is predicted ----

    if not df_kwund.empty:
        precision_list_kw, recall_list_kw = [], []
        
        #df_kwund = all_df[all_df['Truth'] in ['KW_und', 'Background']].copy()
        df_kwund = all_df[all_df['Truth'].isin(['KW_und', 'Background'])].copy()
    
        known_kw_set = set(['SRKW', 'TKW', 'OKW', 'NRKW'])
        y_true = np.ones(len(df_kwund), dtype=bool)  # All positives
        y_score = df_kwund['Top Score'].values
        y_pred_kw = df_kwund['Predicted Class'].isin(known_kw_set)
    
        thresholds = np.linspace(0, 1, 200)
        for thresh in thresholds:
            # Predicted as KW and confident
            TP = ((y_pred_kw) & (y_score >= thresh)).sum()
            FP = ((~y_pred_kw) & (y_score >= thresh)).sum()
            FN = ((y_score < thresh)).sum()  # Missed positive
    
            precision = TP / (TP + FP) if (TP + FP) > 0 else 1.0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    
            precision_list_kw.append(precision)
            recall_list_kw.append(recall)
    
        auc_pr_kwund = auc(recall_list_kw, precision_list_kw)
        pr_data['KW_und (any KW)'] = {
            'thresholds': thresholds,
            'precision': np.array(precision_list_kw),
            'recall': np.array(recall_list_kw)
        }
        auc_pr_dict['KW_und (any KW)'] = auc_pr_kwund
    
        plt.plot(recall_list_kw, precision_list_kw,
                 label='Unknown KW', linestyle='--', color='darkred')
    
    # ---- Wrap up ----
    #mean_ap = np.mean([v for k, v in auc_pr_dict.items() if k != 'KW_und (any KW)'])
    
    # MAP of just relvenvet classes
    mean_ap =(auc_pr_dict['HW']+auc_pr_dict['SRKW']+auc_pr_dict['TKW'])/3

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(titleStr)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    return pr_data, auc_pr_dict, mean_ap



# --- Main Evaluation Block ---
if __name__ == "__main__":
    
    # ---- DCLDE Evaluation ----
    # Folder paths for DCLDE data; keys serve as the truth labels.
    dclde_folders = {
        "Background": r"C:\TempData\DCLDE_HOLDOUT\Background",
        "HW":         r"C:\TempData\DCLDE_HOLDOUT\HW",
        "SRKW":       r"C:\TempData\DCLDE_HOLDOUT\SRKW",
        "TKW":        r"C:\TempData\DCLDE_HOLDOUT\TKW",
    }
        
    
    # ---- Malahat Evaluation ----
    # Folder paths for Malahat data.
    # (For Malahat you may only have a subset of classes; here we use TKW, SRKW, and HW.)
    malahat_folders = {
        "TKW":  r"C:\TempData\Malahat_HOLDOUT\TKW",
        "SRKW": r"C:\TempData\Malahat_HOLDOUT\SRKW",
        "HW":   r"C:\TempData\Malahat_HOLDOUT\HW",
        "Background": r"C:\TempData\Malahat_HOLDOUT\Background",
        "UndKW": r"C:\TempData\Malahat_HOLDOUT\KW_und"
    }    
    
    output_csv = "predictions_output.csv"
    
    #%% Birdnet 01- 2000 clips randomly selected filtered 15khz
    ########################################################################
    # 2000 clips randomly selected filtered 15khz
    
    model_config_01 = {
        "model_path": r"C:\Users\kaity\Documents\GitHub\Ecotype\Experiments\BirdnetOrganized\BirdnetGrid\birdnet01\birdnet01.tflite",
        "label_path": r"C:\Users\kaity\Documents\GitHub\Ecotype\Experiments\BirdnetOrganized\BirdnetGrid\birdnet01\birdnet01_Labels.txt"
    }
    
    

    try:
        eval_dclde_birdnet_01 = pd.read_csv('C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\BirdnetOrganized\BirdnetGrid\birdnet01_DCLDE_eval.csv')
    except:
            # Process all DCLDE datasets into one DataFrame.
            eval_dclde_birdnet_01 = process_multiple_datasets(
                model_config_01["model_path"],
                model_config_01["label_path"], 
                dclde_folders, 
                output_csv='C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\BirdnetOrganized\BirdnetGrid\birdnet01_DCLDE_eval.csv')
            
            eval_dclde_birdnet_01.to_csv('C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\BirdnetOrganized\BirdnetGrid\birdnet01_DCLDE_eval.csv', 
                                         index=False)
    
    
    
    try:
        eval_malahat_birdnet_01 = pd.read_csv('C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\\BirdnetOrganized\birdnet01_Malahat_eval.csv')
        #eval_malahat_birdnet_01 = eval_malahat_birdnet_01[eval_malahat_birdnet_01['Truth'] != 'KW_und'] 

    except:            
        # Process Malahat datasets; note that we adjust the list of classes if OKW is not present.
        eval_malahat_birdnet_01 = process_multiple_datasets(
            model_config_01["model_path"], 
            model_config_01["label_path"], 
            malahat_folders, 
            output_csv='C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\\BirdnetOrganized\\birdnet01_Malahat_eval.csv')  
        eval_malahat_birdnet_01.to_csv('C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\\BirdnetOrganized\\birdnet01_Malahat_eval.csv', index=False)
        
    # Compute custom thresholds for DCLDE (for OKW we use a fixed value)
    hw_cutoff   = compute_threshold(eval_dclde_birdnet_01[eval_dclde_birdnet_01['Truth'] == "HW"],
                                    score_column="HW",  
                                    class_column="Predicted Class",  
                                    title_suffix="Humpback")
    tkw_cutoff  = compute_threshold(eval_dclde_birdnet_01[eval_dclde_birdnet_01['Truth'] == "TKW"], 
                                    score_column="TKW", class_column="Predicted Class", title_suffix="TKW")
    srkw_cutoff = compute_threshold(eval_dclde_birdnet_01[eval_dclde_birdnet_01['Truth'] == "SRKW"], 
                                    score_column="SRKW", class_column="Predicted Class", title_suffix="SRKW")
    custom_thresholds_dclde = {
        "HW": hw_cutoff,
        "TKW": tkw_cutoff,
        "SRKW": srkw_cutoff,
    }
    
    # Compute thresholds for Malahat data
    hw_cutoff_m   = compute_threshold(eval_malahat_birdnet_01[eval_malahat_birdnet_01['Truth'] == "HW"], class_column="Predicted Class", score_column="HW", title_suffix="Malahat HW")
    tkw_cutoff_m  = compute_threshold(eval_malahat_birdnet_01[eval_malahat_birdnet_01['Truth'] == "TKW"], class_column="Predicted Class", score_column="TKW", title_suffix="Malahat TKW")
    srkw_cutoff_m = compute_threshold(eval_malahat_birdnet_01[eval_malahat_birdnet_01['Truth'] == "SRKW"], class_column="Predicted Class", score_column="SRKW", title_suffix="Malahat SRKW")
    custom_thresholds_malahat = {
        "HW": hw_cutoff_m,
        "TKW": tkw_cutoff_m,
        "SRKW": srkw_cutoff_m
    }
    # Evaluate and plot DCLDE performance
    metrics_DCLDE_01 = evaluate_model(eval_dclde_birdnet_01, 
                                      custom_thresholds=custom_thresholds_dclde, 
                                pr_title="DCLDE Precision–Recall Curve 01", 
                                roc_title="DCLDE ROC Curve 01")
    

    # Evaluate and plot Malahat performance
    metrics_malahat_01 = evaluate_model(eval_malahat_birdnet_01, 
                                        custom_thresholds=custom_thresholds_dclde, 
                                  pr_title="Malahat Precision–Recall Curve 01",
                                  roc_title="Malahat ROC Curve 01")
    
    
    
    #%% Birdnet 02- 2000 clips balanced with 100 call type examples 15khz
    ########################################################################
    # In this model we will have a total of 2k annotations for each class, however
    # We will balance across call types with at least 100 in each calltype.
    
    model_config_02 = {
        "model_path": r"C:\Users\kaity\Documents\GitHub\Ecotype\Experiments\BirdnetOrganized\BirdnetGrid\birdnet02\birdnet02.tflite",
        "label_path": r"C:\Users\kaity\Documents\GitHub\Ecotype\Experiments\BirdnetOrganized\BirdnetGrid\birdnet02\birdnet02_Labels.txt"
    }
    
    

    try:
        eval_dclde_birdnet_02 = pd.read_csv('C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\BirdnetOrganized\BirdnetGrid\birdnet02_DCLDE_eval.csv')
    except:
            # Process all DCLDE datasets into one DataFrame.
            eval_dclde_birdnet_02 = process_multiple_datasets(
                model_config_01["model_path"],
                model_config_01["label_path"], 
                dclde_folders, 
                output_csv='C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\BirdnetOrganized\BirdnetGrid\birdnet02_DCLDE_eval.csv')
            
            eval_dclde_birdnet_02.to_csv('C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\BirdnetOrganized\BirdnetGrid\birdnet02_DCLDE_eval.csv', index=False)
    
    
    
    try:
        eval_malahat_birdnet_02 = pd.read_csv('C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\\BirdnetOrganized\birdnet02_Malahat_eval.csv')
        #eval_malahat_birdnet_02 = eval_malahat_birdnet_02[eval_malahat_birdnet_02['Truth'] != 'KW_und'] 

    except:            
        # Process Malahat datasets; note that we adjust the list of classes if OKW is not present.
        eval_malahat_birdnet_02 = process_multiple_datasets(
            model_config_02["model_path"], 
            model_config_02["label_path"], 
            malahat_folders, 
            output_csv='C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\\BirdnetOrganized\\birdnet02_Malahat_eval.csv')  
        eval_malahat_birdnet_02.to_csv('C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\\BirdnetOrganized\\birdnet02_Malahat_eval.csv', index=False)
        
    # Compute custom thresholds for DCLDE (for OKW we use a fixed value)
    hw_cutoff   = compute_threshold(eval_dclde_birdnet_02[eval_dclde_birdnet_02['Truth'] == "HW"],
                                    score_column="HW",  
                                    class_column="Predicted Class",  
                                    title_suffix="Humpback")
    tkw_cutoff  = compute_threshold(eval_dclde_birdnet_02[eval_dclde_birdnet_02['Truth'] == "TKW"], 
                                    score_column="TKW", class_column="Predicted Class", title_suffix="TKW")
    srkw_cutoff = compute_threshold(eval_dclde_birdnet_02[eval_dclde_birdnet_02['Truth'] == "SRKW"], 
                                    score_column="SRKW", class_column="Predicted Class", title_suffix="SRKW")
    custom_thresholds_dclde = {
        "HW": hw_cutoff,
        "TKW": tkw_cutoff,
        "SRKW": srkw_cutoff
    }
    
    # Compute thresholds for Malahat data
    hw_cutoff_m   = compute_threshold(eval_malahat_birdnet_02[eval_malahat_birdnet_02['Truth'] == "HW"], class_column="Predicted Class", score_column="HW", title_suffix="Malahat HW")
    tkw_cutoff_m  = compute_threshold(eval_malahat_birdnet_02[eval_malahat_birdnet_02['Truth'] == "TKW"], class_column="Predicted Class", score_column="TKW", title_suffix="Malahat TKW")
    srkw_cutoff_m = compute_threshold(eval_malahat_birdnet_02[eval_malahat_birdnet_02['Truth'] == "SRKW"], class_column="Predicted Class", score_column="SRKW", title_suffix="Malahat SRKW")
    custom_thresholds_malahat = {
        "HW": hw_cutoff_m,
        "TKW": tkw_cutoff_m,
        "SRKW": srkw_cutoff_m
    }
    # Evaluate and plot DCLDE performance
    metrics_DCLDE_02 = evaluate_model(eval_dclde_birdnet_02, 
                                      custom_thresholds=custom_thresholds_dclde, 
                                pr_title="DCLDE Precision–Recall Curve 02", 
                                roc_title="DCLDE ROC Curve 02")
    

    # Evaluate and plot Malahat performance
    metrics_malahat_02 = evaluate_model(eval_malahat_birdnet_02, 
                                        custom_thresholds=custom_thresholds_dclde, 
                                  pr_title="Malahat Precision–Recall Curve 02",
                                  roc_title="Malahat ROC Curve 02")
    
    #%% Birdnet 03 -2000 clips 100 call types, background 1k undbio/abiotic
    ########################################################################
    # Balanced per class with calltype control for killer whales
    
    model_config_03 = {
        "model_path": r"C:\Users\kaity\Documents\GitHub\Ecotype\Experiments\BirdnetOrganized\BirdnetGrid\birdnet03\birdnet03.tflite",
        "label_path": r"C:\Users\kaity\Documents\GitHub\Ecotype\Experiments\BirdnetOrganized\BirdnetGrid\birdnet03\birdnet03_Labels.txt"
    }
    
    

    try:
        eval_dclde_birdnet_03 = pd.read_csv('C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\BirdnetOrganized\BirdnetGrid\birdnet03_DCLDE_eval.csv')
    except:
            # Process all DCLDE datasets into one DataFrame.
            eval_dclde_birdnet_03 = process_multiple_datasets(
                model_config_03["model_path"],
                model_config_03["label_path"], 
                dclde_folders, 
                output_csv='C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\BirdnetOrganized\BirdnetGrid\birdnet03_DCLDE_eval.csv')
            
            eval_dclde_birdnet_03.to_csv('C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\BirdnetOrganized\BirdnetGrid\birdnet03_DCLDE_eval.csv',
                                         index=False)
    
    
    
    try:
        eval_malahat_birdnet_03 = pd.read_csv('C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\\BirdnetOrganized\birdnet03_Malahat_eval.csv')
      
    except:            
        # Process Malahat datasets; note that we adjust the list of classes if OKW is not present.
        eval_malahat_birdnet_03 = process_multiple_datasets(
            model_config_03["model_path"], 
            model_config_03["label_path"], 
            malahat_folders, 
            output_csv='C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\\BirdnetOrganized\\birdnet03_Malahat_eval.csv', 
            classes=['SRKW','HW','TKW'])  
        eval_malahat_birdnet_03.to_csv('C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\\BirdnetOrganized\\birdnet03_Malahat_eval.csv', index=False)
        
    # Compute custom thresholds for DCLDE (for OKW we use a fixed value)
    hw_cutoff   = compute_threshold(eval_dclde_birdnet_03[eval_dclde_birdnet_03['Truth'] == "HW"],
                                    score_column="HW",  
                                    class_column="Predicted Class",  
                                    title_suffix="Humpback")
    tkw_cutoff  = compute_threshold(eval_dclde_birdnet_03[eval_dclde_birdnet_03['Truth'] == "TKW"], 
                                    score_column="TKW", class_column="Predicted Class", title_suffix="TKW")
    srkw_cutoff = compute_threshold(eval_dclde_birdnet_03[eval_dclde_birdnet_03['Truth'] == "SRKW"], 
                                    score_column="SRKW", class_column="Predicted Class", title_suffix="SRKW")
    custom_thresholds_dclde = {
        "HW": hw_cutoff,
        "TKW": tkw_cutoff,
        "SRKW": srkw_cutoff
    }
    
    # Compute thresholds for Malahat data
    hw_cutoff_m   = compute_threshold(eval_malahat_birdnet_03[eval_malahat_birdnet_03['Truth'] == "HW"], class_column="Predicted Class", score_column="HW", title_suffix="Malahat HW")
    tkw_cutoff_m  = compute_threshold(eval_malahat_birdnet_03[eval_malahat_birdnet_03['Truth'] == "TKW"], class_column="Predicted Class", score_column="TKW", title_suffix="Malahat TKW")
    srkw_cutoff_m = compute_threshold(eval_malahat_birdnet_03[eval_malahat_birdnet_03['Truth'] == "SRKW"], class_column="Predicted Class", score_column="SRKW", title_suffix="Malahat SRKW")
    custom_thresholds_malahat = {
        "HW": hw_cutoff_m,
        "TKW": tkw_cutoff_m,
        "SRKW": srkw_cutoff_m
    }
    # Evaluate and plot DCLDE performance
    metrics_DCLDE_03 = evaluate_model(eval_dclde_birdnet_03, 
                                      custom_thresholds=custom_thresholds_dclde, 
                                pr_title="DCLDE Precision–Recall Curve 03", 
                                roc_title="DCLDE ROC Curve 03")
    

    # Evaluate and plot Malahat performance
    metrics_malahat_03 = evaluate_model(eval_malahat_birdnet_03, 
                                        custom_thresholds=custom_thresholds_dclde, 
                                  pr_title="Malahat Precision–Recall Curve 03",
                                  roc_title="Malahat ROC Curve 03")
    
    #%% Birdnet 04 -3000 clips 200 call replicates, background 1.5k undbio/abiotic 
    ########################################################################
    # 3000 calls per class 200 replicates of each call type
   
    
    model_config_04 = {
        "model_path": r"C:\Users\kaity\Documents\GitHub\Ecotype\Experiments\BirdnetOrganized\BirdnetGrid\birdnet04\birdnet04.tflite",
        "label_path": r"C:\Users\kaity\Documents\GitHub\Ecotype\Experiments\BirdnetOrganized\BirdnetGrid\birdnet04\birdnet04_Labels.txt"
    }
    
    

    try:
        eval_dclde_birdnet_04 = pd.read_csv('C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\BirdnetOrganized\BirdnetGrid\birdnet04_DCLDE_eval.csv')
    except:
            # Process all DCLDE datasets into one DataFrame.
            eval_dclde_birdnet_04 = process_multiple_datasets(
                model_config_04["model_path"],
                model_config_04["label_path"], 
                dclde_folders, 
                output_csv='C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\BirdnetOrganized\BirdnetGrid\birdnet04_DCLDE_eval.csv')
            
            eval_dclde_birdnet_04.to_csv('C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\BirdnetOrganized\BirdnetGrid\birdnet04_DCLDE_eval.csv', index=False)
    
    
    
    try:
        eval_malahat_birdnet_04 = pd.read_csv('C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\\BirdnetOrganized\birdnet04_Malahat_eval.csv')
    
    except:            
        # Process Malahat datasets; note that we adjust the list of classes if OKW is not present.
        eval_malahat_birdnet_04 = process_multiple_datasets(
            model_config_04["model_path"], 
            model_config_04["label_path"], 
            malahat_folders, 
            output_csv='C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\\BirdnetOrganized\\birdnet04_Malahat_eval.csv', 
            classes=['SRKW','HW','TKW'])  
        eval_malahat_birdnet_04.to_csv('C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\\BirdnetOrganized\\birdnet04_Malahat_eval.csv', index=False)
        
    # Compute custom thresholds for DCLDE (for OKW we use a fixed value)
    hw_cutoff   = compute_threshold(eval_dclde_birdnet_04[eval_dclde_birdnet_04['Truth'] == "HW"],
                                    score_column="HW",  
                                    class_column="Predicted Class",  
                                    title_suffix="Humpback")
    tkw_cutoff  = compute_threshold(eval_dclde_birdnet_04[eval_dclde_birdnet_04['Truth'] == "TKW"], 
                                    score_column="TKW", class_column="Predicted Class", title_suffix="TKW")
    srkw_cutoff = compute_threshold(eval_dclde_birdnet_04[eval_dclde_birdnet_04['Truth'] == "SRKW"], 
                                    score_column="SRKW", class_column="Predicted Class", title_suffix="SRKW")
    custom_thresholds_dclde = {
        "HW": hw_cutoff,
        "TKW": tkw_cutoff,
        "SRKW": srkw_cutoff
    }
    
    # Compute thresholds for Malahat data
    hw_cutoff_m   = compute_threshold(eval_malahat_birdnet_04[eval_malahat_birdnet_04['Truth'] == "HW"], class_column="Predicted Class", score_column="HW", title_suffix="Malahat HW")
    tkw_cutoff_m  = compute_threshold(eval_malahat_birdnet_04[eval_malahat_birdnet_04['Truth'] == "TKW"], class_column="Predicted Class", score_column="TKW", title_suffix="Malahat TKW")
    srkw_cutoff_m = compute_threshold(eval_malahat_birdnet_04[eval_malahat_birdnet_04['Truth'] == "SRKW"], class_column="Predicted Class", score_column="SRKW", title_suffix="Malahat SRKW")
    custom_thresholds_malahat = {
        "HW": hw_cutoff_m,
        "TKW": tkw_cutoff_m,
        "SRKW": srkw_cutoff_m
    }
    # Evaluate and plot DCLDE performance
    metrics_DCLDE_04 = evaluate_model(eval_dclde_birdnet_04, 
                                      custom_thresholds=custom_thresholds_dclde, 
                                pr_title="DCLDE Precision–Recall Curve 04", 
                                roc_title="DCLDE ROC Curve 04")
    

    # Evaluate and plot Malahat performance
    metrics_malahat_04 = evaluate_model(eval_malahat_birdnet_04, 
                                        custom_thresholds=custom_thresholds_dclde, 
                                  pr_title="Malahat Precision–Recall Curve 04",
                                  roc_title="Malahat ROC Curve 04")
    #%% Birdnet 05- 6000 clips randomly selected filtered 15khz
    ########################################################################
    # # Same as first birdnet save 6k detections

    
    model_config_05 = {
        "model_path": r"C:\Users\kaity\Documents\GitHub\Ecotype\Experiments\BirdnetOrganized\BirdnetGrid\birdnet05\birdnet05.tflite",
        "label_path": r"C:\Users\kaity\Documents\GitHub\Ecotype\Experiments\BirdnetOrganized\BirdnetGrid\birdnet05\birdnet05_Labels.txt"
    }
    
    

    try:
        eval_dclde_birdnet_05 = pd.read_csv('C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\BirdnetOrganized\BirdnetGrid\birdnet05_DCLDE_eval.csv')
    except:
            # Process all DCLDE datasets into one DataFrame.
            eval_dclde_birdnet_05 = process_multiple_datasets(
                model_config_05["model_path"],
                model_config_05["label_path"], 
                dclde_folders, 
                output_csv='C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\BirdnetOrganized\BirdnetGrid\birdnet05_DCLDE_eval.csv')
            
            eval_dclde_birdnet_04.to_csv('C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\BirdnetOrganized\BirdnetGrid\birdnet05_DCLDE_eval.csv', index=False)
    
    
    
    try:
        eval_malahat_birdnet_05 = pd.read_csv('C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\\BirdnetOrganized\birdnet05_Malahat_eval.csv')
       
    except:            
        # Process Malahat datasets; note that we adjust the list of classes if OKW is not present.
        eval_malahat_birdnet_05 = process_multiple_datasets(
            model_config_05["model_path"], 
            model_config_05["label_path"], 
            malahat_folders, 
            output_csv='C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\\BirdnetOrganized\\birdnet05_Malahat_eval.csv', 
            classes=['SRKW','HW','TKW'])  
        eval_malahat_birdnet_04.to_csv('C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\\BirdnetOrganized\\birdnet05_Malahat_eval.csv', index=False)
        
    # Compute custom thresholds for DCLDE (for OKW we use a fixed value)
    hw_cutoff   = compute_threshold(eval_dclde_birdnet_05[eval_dclde_birdnet_05['Truth'] == "HW"],
                                    score_column="HW",  
                                    class_column="Predicted Class",  
                                    title_suffix="Humpback")
    tkw_cutoff  = compute_threshold(eval_dclde_birdnet_05[eval_dclde_birdnet_05['Truth'] == "TKW"], 
                                    score_column="TKW", class_column="Predicted Class", title_suffix="TKW")
    srkw_cutoff = compute_threshold(eval_dclde_birdnet_05[eval_dclde_birdnet_05['Truth'] == "SRKW"], 
                                    score_column="SRKW", class_column="Predicted Class", title_suffix="SRKW")
    custom_thresholds_dclde = {
        "HW": hw_cutoff,
        "TKW": tkw_cutoff,
        "SRKW": srkw_cutoff
    }
    
    # Compute thresholds for Malahat data
    hw_cutoff_m   = compute_threshold(eval_malahat_birdnet_05[eval_malahat_birdnet_05['Truth'] == "HW"], class_column="Predicted Class", score_column="HW", title_suffix="Malahat HW")
    tkw_cutoff_m  = compute_threshold(eval_malahat_birdnet_05[eval_malahat_birdnet_05['Truth'] == "TKW"], class_column="Predicted Class", score_column="TKW", title_suffix="Malahat TKW")
    srkw_cutoff_m = compute_threshold(eval_malahat_birdnet_05[eval_malahat_birdnet_05['Truth'] == "SRKW"], class_column="Predicted Class", score_column="SRKW", title_suffix="Malahat SRKW")
    custom_thresholds_malahat = {
        "HW": hw_cutoff_m,
        "TKW": tkw_cutoff_m,
        "SRKW": srkw_cutoff_m
    }
    # Evaluate and plot DCLDE performance
    metrics_DCLDE_05 = evaluate_model(eval_dclde_birdnet_05, 
                                      custom_thresholds=custom_thresholds_dclde, 
                                pr_title="DCLDE Precision–Recall Curve 05", 
                                roc_title="DCLDE ROC Curve 05")
    

    # Evaluate and plot Malahat performance
    metrics_malahat_05 = evaluate_model(eval_malahat_birdnet_05, 
                                        custom_thresholds=custom_thresholds_dclde, 
                                  pr_title="Malahat Precision–Recall Curve 05",
                                  roc_title="Malahat ROC Curve 05")

    #%% Birdnet 06- Birdnet 04 filtered 8000hz in the app
    ########################################################################
    # Trained with birdnet04 [3000 calls per class 200 replicates of each call type]
    # data but restricted to 8khz in the app
    
    model_config_06 = {
        "model_path": r"C:\Users\kaity\Documents\GitHub\Ecotype\Experiments\BirdnetOrganized\BirdnetGrid\birdnet06\birdnet06.tflite",
        "label_path": r"C:\Users\kaity\Documents\GitHub\Ecotype\Experiments\BirdnetOrganized\BirdnetGrid\birdnet06\birdnet06_Labels.txt"
    }
    
    

    try:
        eval_dclde_birdnet_06 = pd.read_csv('C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\BirdnetOrganized\BirdnetGrid\birdnet06_DCLDE_eval.csv')
    except:
            # Process all DCLDE datasets into one DataFrame.
            eval_dclde_birdnet_06 = process_multiple_datasets(
                model_config_06["model_path"],
                model_config_06["label_path"], 
                dclde_folders, 
                output_csv='C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\BirdnetOrganized\BirdnetGrid\birdnet06_DCLDE_eval.csv')
            
            eval_dclde_birdnet_04.to_csv('C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\BirdnetOrganized\BirdnetGrid\birdnet06_DCLDE_eval.csv', index=False)
    
    
    
    try:
        eval_malahat_birdnet_06 = pd.read_csv('C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\\BirdnetOrganized\birdnet06_Malahat_eval.csv')
        

    except:            
        # Process Malahat datasets; note that we adjust the list of classes if OKW is not present.
        eval_malahat_birdnet_06 = process_multiple_datasets(
            model_config_06["model_path"], 
            model_config_06["label_path"], 
            malahat_folders, 
            output_csv='C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\\BirdnetOrganized\\birdnet06_Malahat_eval.csv', 
            classes=['SRKW','HW','TKW'])  
        eval_malahat_birdnet_04.to_csv('C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\\BirdnetOrganized\\birdnet06_Malahat_eval.csv', index=False)
        
    # Compute custom thresholds for DCLDE (for OKW we use a fixed value)
    hw_cutoff   = compute_threshold(eval_dclde_birdnet_06[eval_dclde_birdnet_06['Truth'] == "HW"],
                                    score_column="HW",  
                                    class_column="Predicted Class",  
                                    title_suffix="Humpback")
    tkw_cutoff  = compute_threshold(eval_dclde_birdnet_06[eval_dclde_birdnet_06['Truth'] == "TKW"], 
                                    score_column="TKW", class_column="Predicted Class", title_suffix="TKW")
    srkw_cutoff = compute_threshold(eval_dclde_birdnet_06[eval_dclde_birdnet_06['Truth'] == "SRKW"], 
                                    score_column="SRKW", class_column="Predicted Class", title_suffix="SRKW")
    custom_thresholds_dclde = {
        "HW": hw_cutoff,
        "TKW": tkw_cutoff,
        "SRKW": srkw_cutoff
    }
    
    # Compute thresholds for Malahat data
    hw_cutoff_m   = compute_threshold(eval_malahat_birdnet_06[eval_malahat_birdnet_06['Truth'] == "HW"], class_column="Predicted Class", score_column="HW", title_suffix="Malahat HW")
    tkw_cutoff_m  = compute_threshold(eval_malahat_birdnet_06[eval_malahat_birdnet_06['Truth'] == "TKW"], class_column="Predicted Class", score_column="TKW", title_suffix="Malahat TKW")
    srkw_cutoff_m = compute_threshold(eval_malahat_birdnet_06[eval_malahat_birdnet_06['Truth'] == "SRKW"], class_column="Predicted Class", score_column="SRKW", title_suffix="Malahat SRKW")
    custom_thresholds_malahat = {
        "HW": hw_cutoff_m,
        "TKW": tkw_cutoff_m,
        "SRKW": srkw_cutoff_m
    }
    # Evaluate and plot DCLDE performance
    metrics_DCLDE_06 = evaluate_model(eval_dclde_birdnet_06, 
                                      custom_thresholds=custom_thresholds_dclde, 
                                pr_title="DCLDE Precision–Recall Curve 06", 
                                roc_title="DCLDE ROC Curve 06")
    

    # Evaluate and plot Malahat performance
    metrics_malahat_06 = evaluate_model(eval_malahat_birdnet_06, 
                                        custom_thresholds=custom_thresholds_dclde, 
                                  pr_title="Malahat Precision–Recall Curve 06",
                                  roc_title="Malahat ROC Curve 06")
    #%% Birdnet 07- 4.5k examples, shifting for augmentation OKW, NRKW
    ########################################################################
    # Run at 15khz in birdnet app
    
    
    model_config_07 = {
        "model_path": r"C:\Users\kaity\Documents\GitHub\Ecotype\Experiments\BirdnetOrganized\BirdnetGrid\birdnet07\birdnet07.tflite",
        "label_path": r"C:\Users\kaity\Documents\GitHub\Ecotype\Experiments\BirdnetOrganized\BirdnetGrid\birdnet07\birdnet07_Labels.txt"
    }
    
    
    
    try:
        eval_dclde_birdnet_07 = pd.read_csv('C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\BirdnetOrganized\BirdnetGrid\birdnet07_DCLDE_eval.csv')
    except:
            # Process all DCLDE datasets into one DataFrame.
            eval_dclde_birdnet_07 = process_multiple_datasets(
                model_config_07["model_path"],
                model_config_07["label_path"], 
                dclde_folders, 
                output_csv='C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\BirdnetOrganized\BirdnetGrid\birdnet07_DCLDE_eval.csv')
            
            eval_dclde_birdnet_07.to_csv('C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\BirdnetOrganized\BirdnetGrid\birdnet07_DCLDE_eval.csv', index=False)
    
    
    
    try:
        eval_malahat_birdnet_07 = pd.read_csv('C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\\BirdnetOrganized\birdnet07_Malahat_eval.csv')
       
    except:            
        # Process Malahat datasets; note that we adjust the list of classes if OKW is not present.
        eval_malahat_birdnet_07 = process_multiple_datasets(
            model_config_07["model_path"], 
            model_config_07["label_path"], 
            malahat_folders, 
            output_csv='C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\\BirdnetOrganized\\birdnet07_Malahat_eval.csv', 
            classes=['SRKW','HW','TKW'])  
        eval_malahat_birdnet_07.to_csv('C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\\BirdnetOrganized\\birdnet07_Malahat_eval.csv', index=False)
        
        
        
    # Compute custom thresholds for DCLDE (for OKW we use a fixed value)
    hw_cutoff   = compute_threshold(eval_dclde_birdnet_07[eval_dclde_birdnet_07['Truth'] == "HW"],
                                    score_column="HW",  
                                    class_column="Predicted Class",  
                                    title_suffix="Humpback")
    tkw_cutoff  = compute_threshold(eval_dclde_birdnet_07[eval_dclde_birdnet_07['Truth'] == "TKW"], 
                                    score_column="TKW", class_column="Predicted Class", title_suffix="TKW")
    srkw_cutoff = compute_threshold(eval_dclde_birdnet_07[eval_dclde_birdnet_07['Truth'] == "SRKW"], 
                                    score_column="SRKW", class_column="Predicted Class", title_suffix="SRKW")
    custom_thresholds_dclde = {
        "HW": hw_cutoff,
        "TKW": tkw_cutoff,
        "SRKW": srkw_cutoff
    }
    
    # Compute thresholds for Malahat data
    hw_cutoff_m   = compute_threshold(eval_malahat_birdnet_07[eval_malahat_birdnet_07['Truth'] == "HW"], class_column="Predicted Class", score_column="HW", title_suffix="Malahat HW")
    tkw_cutoff_m  = compute_threshold(eval_malahat_birdnet_07[eval_malahat_birdnet_07['Truth'] == "TKW"], class_column="Predicted Class", score_column="TKW", title_suffix="Malahat TKW")
    srkw_cutoff_m = compute_threshold(eval_malahat_birdnet_07[eval_malahat_birdnet_07['Truth'] == "SRKW"], class_column="Predicted Class", score_column="SRKW", title_suffix="Malahat SRKW")
    custom_thresholds_malahat = {
        "HW": hw_cutoff_m,
        "TKW": tkw_cutoff_m,
        "SRKW": srkw_cutoff_m
    }
    # Evaluate and plot DCLDE performance
    metrics_DCLDE_07 = evaluate_model(eval_dclde_birdnet_07, 
                                      custom_thresholds=custom_thresholds_dclde, 
                                pr_title="DCLDE Precision–Recall Curve 07", 
                                roc_title="DCLDE ROC Curve 07")
    
    
    # Evaluate and plot Malahat performance
    metrics_malahat_07 = evaluate_model(eval_malahat_birdnet_07, 
                                        custom_thresholds=custom_thresholds_dclde, 
                                  pr_title="Malahat Precision–Recall Curve 07",
                                  roc_title="Malahat ROC Curve 07")
 
    
    
    
    #%% Birdnet 08- birdnet 07 but Ecotype-balanced with augmentation and stratified background
    
    # Filters Ecotype directly and excludes ambiguous call types manually
    # Explicit mapping of call types (N01, S03, etc.) to each ecotype using a custom assign_calltype() function
    # Augmentation Applied per ecotype using ecotype-specific calltype lists, then pads with unannotated ecotype data
    # Stratified across Provider before sampling 4600 examples each
    # Filters out a detailed list of call types (Buzz, Whistle, Unk, etc.) and excludes annotations with a ?
    # Augments every mapped calltype to ensure representation per ecotype
    
    
    
    model_config_08 = {
        "model_path": r"C:\Users\kaity\Documents\GitHub\Ecotype\Experiments\BirdnetOrganized\BirdnetGrid\birdnet08\birdnet08.tflite",
        "label_path": r"C:\Users\kaity\Documents\GitHub\Ecotype\Experiments\BirdnetOrganized\BirdnetGrid\birdnet08\birdnet08_Labels.txt"
    }
    
    
    
    try:
        eval_dclde_birdnet_08 = pd.read_csv('C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\BirdnetOrganized\BirdnetGrid\birdnet08_DCLDE_eval.csv')
    except:
            # Process all DCLDE datasets into one DataFrame.
            eval_dclde_birdnet_08 = process_multiple_datasets(
                model_config_08["model_path"],
                model_config_08["label_path"], 
                dclde_folders, 
                output_csv='C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\BirdnetOrganized\BirdnetGrid\birdnet08_DCLDE_eval.csv')
            
            eval_dclde_birdnet_08.to_csv('C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\BirdnetOrganized\BirdnetGrid\birdnet08_DCLDE_eval.csv', index=False)
    
    
    
    try:
        eval_malahat_birdnet_08 = pd.read_csv('C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\\BirdnetOrganized\birdnet08_Malahat_eval.csv')
        # Exclude unidentified KW detections for test
        malahat_folders
    except:            
        # Process Malahat datasets; note that we adjust the list of classes if OKW is not present.
        eval_malahat_birdnet_08 = process_multiple_datasets(
            model_config_08["model_path"], 
            model_config_08["label_path"], 
            malahat_folders, 
            output_csv='C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\\BirdnetOrganized\\birdnet08_Malahat_eval.csv', 
            classes=['SRKW','HW','TKW'])  
        eval_malahat_birdnet_08.to_csv('C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\\BirdnetOrganized\\birdnet08_Malahat_eval.csv', index=False)
        
        
        
    # Compute custom thresholds for DCLDE (for OKW we use a fixed value)
    hw_cutoff   = compute_threshold(eval_dclde_birdnet_08[eval_dclde_birdnet_08['Truth'] == "HW"],
                                    score_column="HW",  
                                    class_column="Predicted Class",  
                                    title_suffix="Humpback")
    tkw_cutoff  = compute_threshold(eval_dclde_birdnet_08[eval_dclde_birdnet_08['Truth'] == "TKW"], 
                                    score_column="TKW", class_column="Predicted Class", title_suffix="TKW")
    srkw_cutoff = compute_threshold(eval_dclde_birdnet_08[eval_dclde_birdnet_08['Truth'] == "SRKW"], 
                                    score_column="SRKW", class_column="Predicted Class", title_suffix="SRKW")
    custom_thresholds_dclde = {
        "HW": hw_cutoff,
        "TKW": tkw_cutoff,
        "SRKW": srkw_cutoff
    }
    
    # Compute thresholds for Malahat data
    hw_cutoff_m   = compute_threshold(eval_malahat_birdnet_08[eval_malahat_birdnet_08['Truth'] == "HW"], 
                                      class_column="Predicted Class", score_column="HW", title_suffix="Malahat HW")
    tkw_cutoff_m  = compute_threshold(eval_malahat_birdnet_08[eval_malahat_birdnet_08['Truth'] == "TKW"], 
                                      class_column="Predicted Class", score_column="TKW", title_suffix="Malahat TKW")
    srkw_cutoff_m = compute_threshold(eval_malahat_birdnet_08[eval_malahat_birdnet_08['Truth'] == "SRKW"], 
                                      class_column="Predicted Class", score_column="SRKW", title_suffix="Malahat SRKW")
    custom_thresholds_malahat = {
        "HW": hw_cutoff_m,
        "TKW": tkw_cutoff_m,
        "SRKW": srkw_cutoff_m
    }
    # Evaluate and plot DCLDE performance
    metrics_DCLDE_08 = evaluate_model(eval_dclde_birdnet_08, 
                                      custom_thresholds=custom_thresholds_malahat, 
                                pr_title="DCLDE Precision–Recall Curve 08", 
                                roc_title="DCLDE ROC Curve 08")
    
    
    # Evaluate and plot Malahat performance
    metrics_malahat_08 = evaluate_model(eval_malahat_birdnet_08, 
                                        custom_thresholds=custom_thresholds_dclde, 
                                  pr_title="Malahat Precision–Recall Curve 08",
                                  roc_title="Malahat ROC Curve 08")
 
    #%% Birdnet 09- birdnet 08 2k files and only HW, SRKW, TKW
    
    # Filters Ecotype directly and excludes ambiguous call types manually
    # Explicit mapping of call types (N01, S03, etc.) to each ecotype using a custom assign_calltype() function
    # Augmentation Applied per ecotype using ecotype-specific calltype lists, then pads with unannotated ecotype data
    # Stratified across Provider before sampling 4600 examples each
    # Filters out a detailed list of call types (Buzz, Whistle, Unk, etc.) and excludes annotations with a ?
    # Augments every mapped calltype to ensure representation per ecotype
    
    
    
    model_config_09 = {
        "model_path": r"C:\Users\kaity\Documents\GitHub\Ecotype\Experiments\BirdnetOrganized\BirdnetGrid\birdnet09\birdnet09.tflite",
        "label_path": r"C:\Users\kaity\Documents\GitHub\Ecotype\Experiments\BirdnetOrganized\BirdnetGrid\birdnet09\birdnet09_Labels.txt"
    }
    
    
    
    try:
        eval_dclde_birdnet_09 = pd.read_csv('C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\BirdnetOrganized\BirdnetGrid\birdnet09_DCLDE_eval.csv')
    except:
            # Process all DCLDE datasets into one DataFrame.
            eval_dclde_birdnet_09 = process_multiple_datasets(
                model_config_09["model_path"],
                model_config_09["label_path"], 
                dclde_folders, 
                output_csv='C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\BirdnetOrganized\BirdnetGrid\birdnet09_DCLDE_eval.csv')
            
            eval_dclde_birdnet_09.to_csv('C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\BirdnetOrganized\BirdnetGrid\birdnet09_DCLDE_eval.csv', index=False)
    
    
    
    try:
        eval_malahat_birdnet_09 = pd.read_csv('C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\\BirdnetOrganized\birdnet09_Malahat_eval.csv')
        # Exclude unidentified KW detections for test
        
    except:            
        # Process Malahat datasets; note that we adjust the list of classes if OKW is not present.
        eval_malahat_birdnet_09 = process_multiple_datasets(
            model_config_09["model_path"], 
            model_config_09["label_path"], 
            malahat_folders, 
            output_csv='C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\\BirdnetOrganized\\birdnet09_Malahat_eval.csv', 
            classes=['SRKW','HW','TKW'])  
        eval_malahat_birdnet_09.to_csv('C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\\BirdnetOrganized\\birdnet09_Malahat_eval.csv', index=False)
        
        
        
    # Compute custom thresholds for DCLDE (for OKW we use a fixed value)
    hw_cutoff   = compute_threshold(eval_dclde_birdnet_09[eval_dclde_birdnet_09['Truth'] == "HW"],
                                    score_column="HW",  
                                    class_column="Predicted Class",  
                                    title_suffix="Humpback")
    tkw_cutoff  = compute_threshold(eval_dclde_birdnet_09[eval_dclde_birdnet_09['Truth'] == "TKW"], 
                                    score_column="TKW", class_column="Predicted Class", title_suffix="TKW")
    srkw_cutoff = compute_threshold(eval_dclde_birdnet_09[eval_dclde_birdnet_09['Truth'] == "SRKW"], 
                                    score_column="SRKW", class_column="Predicted Class", title_suffix="SRKW")
    custom_thresholds_dclde = {
        "HW": hw_cutoff,
        "TKW": tkw_cutoff,
        "SRKW": srkw_cutoff
    }
    
    # Compute thresholds for Malahat data
    hw_cutoff_m   = compute_threshold(eval_malahat_birdnet_09[eval_malahat_birdnet_09['Truth'] == "HW"], 
                                      class_column="Predicted Class", score_column="HW", title_suffix="Malahat HW")
    tkw_cutoff_m  = compute_threshold(eval_malahat_birdnet_09[eval_malahat_birdnet_09['Truth'] == "TKW"], 
                                      class_column="Predicted Class", score_column="TKW", title_suffix="Malahat TKW")
    srkw_cutoff_m = compute_threshold(eval_malahat_birdnet_09[eval_malahat_birdnet_09['Truth'] == "SRKW"], 
                                      class_column="Predicted Class", score_column="SRKW", title_suffix="Malahat SRKW")
    custom_thresholds_malahat = {
        "HW": hw_cutoff_m,
        "TKW": tkw_cutoff_m,
        "SRKW": srkw_cutoff_m
    }
    # Evaluate and plot DCLDE performance
    metrics_DCLDE_09 = evaluate_model(eval_dclde_birdnet_09, 
                                      custom_thresholds=custom_thresholds_dclde, 
                                pr_title="DCLDE Precision–Recall Curve 09", 
                                roc_title="DCLDE ROC Curve 09")
    
    
    # Evaluate and plot Malahat performance
    metrics_malahat_09 = evaluate_model(eval_malahat_birdnet_09, 
                                        custom_thresholds=custom_thresholds_dclde, 
                                  pr_title="Malahat Precision–Recall Curve 09",
                                  roc_title="Malahat ROC Curve 09")    
    

    #%% Combine metrics for sanity

    modelNames = address = ['birdNET_01','birdNET_02',
                            'birdNET_03', 'birdNET_04',
                            'birdNET_05', 'birdNET_06',
                            'birdNET_07', 'birdNet_08', 'birdNet_09']


    AUCDCLDE = pd.DataFrame([
        metrics_DCLDE_01['AUC'],
        metrics_DCLDE_02['AUC'],
        metrics_DCLDE_03['AUC'],
                  metrics_DCLDE_04['AUC'],
                  metrics_DCLDE_05['AUC'],
                  metrics_DCLDE_06['AUC'],
                  metrics_malahat_07['AUC'],
                  metrics_malahat_08['AUC'],
                  metrics_malahat_09['AUC']]).fillna(0)
    AUCDCLDE['Model'] =modelNames


    MAP_DCLDE = pd.DataFrame([metrics_DCLDE_01['MAP'],
                              metrics_DCLDE_02['MAP'],
                              metrics_DCLDE_03['MAP'],
                  metrics_DCLDE_04['MAP'],
                  metrics_DCLDE_05['MAP'],
                  metrics_DCLDE_06['MAP'],
                  metrics_DCLDE_07['MAP'],
                  metrics_DCLDE_08['MAP'], 
                  metrics_DCLDE_09['MAP'] ]).fillna(0)
    MAP_DCLDE['Model'] =modelNames
    
    MCC_DCLDE = pd.DataFrame([metrics_DCLDE_01['mcc_srkw_tkw'],
                              metrics_DCLDE_02['mcc_srkw_tkw'],
                              metrics_DCLDE_03['mcc_srkw_tkw'],
                  metrics_DCLDE_04['mcc_srkw_tkw'],
                  metrics_DCLDE_05['mcc_srkw_tkw'],
                  metrics_DCLDE_06['mcc_srkw_tkw'],
                  metrics_DCLDE_07['mcc_srkw_tkw'],
                  metrics_DCLDE_08['mcc_srkw_tkw'], 
                  metrics_DCLDE_09['mcc_srkw_tkw']]).fillna(0)
    MCC_DCLDE['Model'] =modelNames
    
    MCC_DCLDE_HW = pd.DataFrame([metrics_DCLDE_01['mcc_srkw_hw'],
                              metrics_DCLDE_02['mcc_srkw_hw'],
                              metrics_DCLDE_03['mcc_srkw_hw'],
                  metrics_DCLDE_04['mcc_srkw_hw'],
                  metrics_DCLDE_05['mcc_srkw_hw'],
                  metrics_DCLDE_06['mcc_srkw_hw'],
                  metrics_DCLDE_07['mcc_srkw_hw'],
                  metrics_DCLDE_08['mcc_srkw_hw'], 
                  metrics_DCLDE_09['mcc_srkw_hw']]).fillna(0)
    MCC_DCLDE_HW['Model'] =modelNames
    



    AUCMalahat= pd.DataFrame([
        metrics_malahat_01['AUC'],
        metrics_malahat_02['AUC'],
        metrics_malahat_03['AUC'],
                  metrics_malahat_04['AUC'],
                  metrics_malahat_05['AUC'],
                  metrics_malahat_06['AUC'],
                  metrics_malahat_07['AUC'], 
                  metrics_malahat_08['AUC'],
                  metrics_malahat_09['AUC']]).fillna(0)
    
    AUCMalahat['Model'] =modelNames

    MAP_Malahat= pd.DataFrame([
        metrics_malahat_01['MAP'],
        metrics_malahat_02['MAP'],
        metrics_malahat_03['MAP'],
                  metrics_malahat_04['MAP'],
                  metrics_malahat_05['MAP'],
                  metrics_malahat_06['MAP'],
                  metrics_malahat_07['MAP'], 
                  metrics_malahat_08['MAP'],
                  metrics_malahat_09['MAP']]).fillna(0)
    MAP_Malahat['Model'] =modelNames
    
    MCC_Malahat= pd.DataFrame([
        metrics_malahat_01['mcc_srkw_tkw'],
        metrics_malahat_02['mcc_srkw_tkw'],
        metrics_malahat_03['mcc_srkw_tkw'],
                  metrics_malahat_04['mcc_srkw_tkw'],
                  metrics_malahat_05['mcc_srkw_tkw'],
                  metrics_malahat_06['mcc_srkw_tkw'],
                  metrics_malahat_07['mcc_srkw_tkw'], 
                  metrics_malahat_08['mcc_srkw_tkw'],
                  metrics_malahat_09['mcc_srkw_tkw']]).fillna(0)
    MCC_Malahat['Model'] =modelNames
    
    MCC_Malahat_HW= pd.DataFrame([
        metrics_malahat_01['mcc_srkw_hw'],
        metrics_malahat_02['mcc_srkw_hw'],
        metrics_malahat_03['mcc_srkw_hw'],
                  metrics_malahat_04['mcc_srkw_hw'],
                  metrics_malahat_05['mcc_srkw_hw'],
                  metrics_malahat_06['mcc_srkw_hw'],
                  metrics_malahat_07['mcc_srkw_hw'], 
                  metrics_malahat_08['mcc_srkw_hw'],
                  metrics_malahat_09['mcc_srkw_hw']]).fillna(0)
    MCC_Malahat_HW['Model'] =modelNames

#%% Plot the precision recall curves

# Malahat
compare_pr_curves(
    metrics_list=[metrics_malahat_01, metrics_malahat_02,metrics_malahat_03,
                  metrics_malahat_04,metrics_malahat_05,metrics_malahat_06,
                  metrics_malahat_07,metrics_malahat_08,
                  metrics_malahat_09],
    model_labels=["MALAHAT_01", "MALAHAT_02","MALAHAT_03","MALAHAT_04",
                  "MALAHAT_05","MALAHAT_06","MALAHAT_07","MALAHAT_08",
                  "MALAHAT_09"],
    target_class="SRKW")

compare_pr_curves(
    metrics_list=[metrics_malahat_01, metrics_malahat_02,metrics_malahat_03,
                  metrics_malahat_04,metrics_malahat_05,metrics_malahat_06,
                  metrics_malahat_07,metrics_malahat_08,
                  metrics_malahat_09],
    model_labels=["MALAHAT_01", "MALAHAT_02","MALAHAT_03","MALAHAT_04",
                  "MALAHAT_05","MALAHAT_06","MALAHAT_07","MALAHAT_08","MALAHAT_09"],
    target_class="TKW"
)

compare_pr_curves(
    metrics_list=[metrics_malahat_01, metrics_malahat_02,metrics_malahat_03,
                  metrics_malahat_04,metrics_malahat_05,metrics_malahat_06,
                  metrics_malahat_07,metrics_malahat_08,
                  metrics_malahat_09],
    model_labels=["MALAHAT_01", "MALAHAT_02","MALAHAT_03","MALAHAT_04",
                  "MALAHAT_05","MALAHAT_06","MALAHAT_07","MALAHAT_08","MALAHAT_09"],
    target_class="HW"
)

# DCLDE 
compare_pr_curves(
    metrics_list=[metrics_DCLDE_01, metrics_DCLDE_02,metrics_DCLDE_03,
                  metrics_DCLDE_04,metrics_DCLDE_05,metrics_DCLDE_06,
                  metrics_DCLDE_07,metrics_DCLDE_08,
                  metrics_DCLDE_09],
    model_labels=["DCLDE_01", "DCLDE_02","DCLDE_03","DCLDE_04",
                  "DCLDE_05","DCLDE_06","DCLDE_07","DCLDE_08",
                  "DCLDE_09", "DCLDE_10", "DCLDE_11"],
    target_class="SRKW")

compare_pr_curves(
    metrics_list=[metrics_DCLDE_01, metrics_DCLDE_02,metrics_DCLDE_03,
                  metrics_DCLDE_04,metrics_DCLDE_05,metrics_DCLDE_06,
                  metrics_DCLDE_07,metrics_DCLDE_08,
                  metrics_DCLDE_09],
    model_labels=["DCLDE_01", "DCLDE_02","DCLDE_03","DCLDE_04",
                  "DCLDE_05","DCLDE_06","DCLDE_07","DCLDE_08","DCLDE_09"],
    target_class="TKW"
)

compare_pr_curves(
    metrics_list=[metrics_DCLDE_01, metrics_DCLDE_02,metrics_DCLDE_03,
                  metrics_DCLDE_04,metrics_DCLDE_05,metrics_DCLDE_06,
                  metrics_DCLDE_07,metrics_DCLDE_08,
                  metrics_DCLDE_09],
    model_labels=["DCLDE_01", "DCLDE_02","DCLDE_03","DCLDE_04",
                  "DCLDE_05","DCLDE_06","DCLDE_07","DCLDE_08","DCLDE_09"],
    target_class="HW"
)

metrics_DCLDE_01['cm']
metrics_DCLDE_02['cm']
metrics_DCLDE_03['cm']
metrics_DCLDE_04['cm']
metrics_DCLDE_05['cm']
metrics_DCLDE_06['cm']
metrics_DCLDE_07['cm']
metrics_DCLDE_08['cm']
metrics_DCLDE_09['cm']


#%% Single panel plot
def plot_mcc_at_p90_from_metrics(
    metrics_list, model_labels, 
    title="MCC halves at logistic P90 thresholds",
    xlim=None, ylim=None, auto_zoom=True,
    pad=0.03, min_span=0.15,
    # size scaling
    size_method="log",       # "log" | "linear" | "sqrt"
    log_k=20.0,              # keep reasonably large; compresses big MCCs
    r_min=0.012, r_max=0.045,# base radii (axis units) before global scaling
    radius_scale=1.0,        # <--- NEW: multiply all radii by this (e.g., 0.25)
    legend_levels=(0.2, 0.5, 0.8),
    label_bg=True
):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Wedge, Circle

    # --- collect data (unchanged) ---
    xs, ys, mcc_l, mcc_r, labs = [], [], [], [], []
    for lab, met in zip(model_labels, metrics_list):
        recs = met.get('recall_at_p90', {})
        y = float(recs.get('SRKW', np.nan))
        x = float(recs.get('TKW',  np.nan))
        if np.isnan(x) or np.isnan(y):
            continue
        xs.append(x); ys.append(y)
        mcc_l.append(float(np.nan_to_num(met.get('mcc_srkw_tkw', np.nan), nan=0.0)))
        mcc_r.append(float(np.nan_to_num(met.get('mcc_srkw_hw',  np.nan), nan=0.0)))
        labs.append(lab)

    xs = np.array(xs, float); ys = np.array(ys, float)
    mcc_l = np.clip(np.array(mcc_l, float), 0.0, 1.0)
    mcc_r = np.clip(np.array(mcc_r, float), 0.0, 1.0)

    # --- size mapping (no hacks) ---
    def _scale(m):
        if size_method == "linear": return m
        if size_method == "sqrt":   return np.sqrt(m)
        # "log" using log1p curve
        return np.log1p(log_k * m) / np.log1p(log_k)

    sL = _scale(mcc_l); sR = _scale(mcc_r)
    rL = (r_min + (r_max - r_min) * sL) * radius_scale
    rR = (r_min + (r_max - r_min) * sR) * radius_scale

    fig, ax = plt.subplots(figsize=(8, 7))

    # --- smart limits (unchanged) ---
    if auto_zoom and xs.size and ys.size and (xlim is None or ylim is None):
        x_min = float(np.nanmin(xs)); x_max = float(np.nanmax(xs))
        y_min = float(np.nanmin(ys)); y_max = float(np.nanmax(ys))
        if (x_max - x_min) < min_span:
            mid = 0.5*(x_min + x_max); x_min, x_max = mid - 0.5*min_span, mid + 0.5*min_span
        if (y_max - y_min) < min_span:
            mid = 0.5*(y_min + y_max); y_min, y_max = mid - 0.5*min_span, mid + 0.5*min_span
        x_min, x_max = max(0.0, x_min - pad), min(1.0, x_max + pad)
        y_min, y_max = max(0.0, y_min - pad), min(1.0, y_max + pad)
        ax.set_xlim(xlim if xlim else (x_min, x_max))
        ax.set_ylim(ylim if ylim else (y_min, y_max))
    else:
        ax.set_xlim(xlim if xlim else (0, 1))
        ax.set_ylim(ylim if ylim else (0, 1))

    ax.set_ylabel("SRKW recall @ logistic P90")
    ax.set_xlabel("TKW recall @ logistic P90")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    # --- wedges ---
    for x, y, rl, rr, lab in zip(xs, ys, rL, rR, labs):
        left_half  = Wedge((x, y), rl,  90, 270, facecolor="#1f77b4", edgecolor="black", linewidth=0.6, alpha=0.9)
        right_half = Wedge((x, y), rr, -90,  90, facecolor="#2ca02c", edgecolor="black", linewidth=0.6, alpha=0.9)
        ax.add_patch(left_half); ax.add_patch(right_half)
        if label_bg:
            ax.text(x + 0.012, y + 0.012, lab, fontsize=9, ha="left", va="bottom",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.6, pad=1.5))
        else:
            ax.text(x + 0.012, y + 0.012, lab, fontsize=9, ha="left", va="bottom")

    # --- half-color legend (unchanged) ---
    sample_left  = Wedge((1.05, 1.05), 0.04,  90, 270, facecolor="#1f77b4", edgecolor="black")
    sample_right = Wedge((1.05, 1.05), 0.04, -90,  90, facecolor="#2ca02c", edgecolor="black")
    ax.legend([sample_left, sample_right],
              ["MCC (SRKW | TKW)", "MCC (SRKW | HW)"],
              loc="lower right", frameon=True)

    # --- size legend (scaled consistently) ---
    # --- size legend (MCC -> radius), scaled consistently with radius_scale ---
    x0, y0, dy = 0.02, 0.96, 0.08
    ax.text(x0, y0, "Size ∝ MCC", transform=ax.transAxes, fontsize=9, va="top")
    
    for i, m in enumerate(legend_levels):
        # use the SAME scaling pipeline as the wedges
        s = _scale(np.array([m]))[0]
        r = (r_min + (r_max - r_min) * s) * radius_scale
    
        # draw legend circles in axes coords; normalize by the *scaled* max radius
        ax_rad = 0.04 * (r / (r_max * radius_scale))
        cx, cy = x0 + 0.05, y0 - (i + 1) * dy
    
        circ = Circle((cx, cy), ax_rad, transform=ax.transAxes,
                      facecolor="#999999", edgecolor="black", alpha=0.6)
        ax.add_patch(circ)
        ax.text(cx + 0.06, cy, f"MCC {m:.1f}", transform=ax.transAxes,
                va="center", fontsize=9)


    plt.tight_layout()
    plt.show()


#%%

# Malahat panel
plot_mcc_at_p90_from_metrics(
    metrics_list=[metrics_malahat_01, metrics_malahat_02, metrics_malahat_03,
                  metrics_malahat_04, metrics_malahat_05, metrics_malahat_06,
                  metrics_malahat_07, metrics_malahat_08, metrics_malahat_09],
    size_method= 'linear',
    radius_scale = 0.3,
    model_labels=["BN01","BN02","BN03","BN04","BN05","BN06","BN07","BN08","BN09"],
    title="Malahat — MCC halves sized by MCC, positioned at recall@logistic P90"
)


# Malahat panel
plot_mcc_at_p90_from_metrics(
    metrics_list=[metrics_DCLDE_01, metrics_DCLDE_02, metrics_DCLDE_03,
                  metrics_DCLDE_04, metrics_DCLDE_05, metrics_DCLDE_06,
                  metrics_DCLDE_07, metrics_DCLDE_08, metrics_DCLDE_09],
    size_method= 'linear',
    radius_scale = 0.3,
    model_labels=["BN01","BN02","BN03","BN04","BN05","BN06","BN07","BN08","BN09"],
    title="DCLDE — MCC halves sized by MCC, positioned at recall@logistic P90"
)


