import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  
import joblib  # For saving the model
import warnings
# Suppress warnings for cleaner output
import optuna
from optuna.samplers import TPESampler
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from configs.config_cytomodel import config
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE  # SMOTE for oversampling
from imblearn.pipeline import Pipeline as ImbPipeline  # Pipeline compatible with imbalanced-learn
warnings.filterwarnings('ignore')
from svm_config import *
from sklearn.metrics import (
    roc_curve, roc_auc_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score, precision_recall_curve,
    average_precision_score
)
import shap
import matplotlib.colors as mcolors

skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
def hyper_param_svm(X_train, y_train,n_trials = 200):
    
    def objective(trial):

        kernel = trial.suggest_categorical('kernel', ['poly'])
        C = trial.suggest_loguniform('C', 1e-3, 1e3)
        if kernel in ['poly', 'rbf', 'sigmoid']:
            gamma = trial.suggest_loguniform('gamma', 1e-4, 1e1)
        else:
            gamma = 'scale'  # for 'linear' kernel, gamma is not used
        # Only tune degree for polynomial kernel
        if kernel == 'poly':
            degree = trial.suggest_int('degree', 2, 5)
        else:
            degree = 3  # default for other kernels (won't matter if kernel != 'poly')
        # coef0 is used by 'poly' and 'sigmoid'
        if kernel in ['poly', 'sigmoid']:
            coef0 = trial.suggest_uniform('coef0', 0.0, 1.0)
        else:
            coef0 = 0.0  # default, not used by 'linear'/'rbf'
        # tune class_weight (especially if data is imbalanced)
        class_weight = trial.suggest_categorical('class_weight', [None, 'balanced'])
        auc_scores = []
        for train_idx, valid_idx in skf.split(X_train, y_train):
            X_train_cv, X_valid_cv = X_train[train_idx], X_train[valid_idx]
            y_train_cv, y_valid_cv = y_train[train_idx], y_train[valid_idx]
            # --- 2) Scale the data ---
            scaler = StandardScaler()
            X_train_cv_scaled = scaler.fit_transform(X_train_cv)
            X_valid_cv_scaled = scaler.transform(X_valid_cv)
    
            # --- 3) Build SVC with suggested params ---
            svm_model = SVC(
                kernel=kernel,
                C=C,
                gamma=gamma,
                degree=degree,
                coef0=coef0,
                class_weight=class_weight,
                probability=True,  # for predict_proba => needed to compute AUC
                random_state=42,
                # tol, shrinking, max_iter could also be tuned if desired
            )
    
            # --- 4) Train and evaluate ---
            svm_model.fit(X_train_cv_scaled, y_train_cv)
            y_pred_proba = svm_model.predict_proba(X_valid_cv_scaled)[:, 1]
            fold_auc = roc_auc_score(y_valid_cv, y_pred_proba)
            auc_scores.append(fold_auc)
        # print(np.mean(auc_scores))
        # Return the mean AUC across folds
        return np.mean(auc_scores)



    
    # --- 5) Create and run the study ---
    study = optuna.create_study(direction='maximize', study_name="SVM Optimization")
    study.optimize(objective, n_trials=n_trials, timeout=None)  # Adjust n_trials/timeout as needed
    
    # --- 6) Print best parameters ---
    best_trial = study.best_trial
    print("Best trial AUC:", best_trial.value)
    print("Best trial parameters:")
    for key, val in best_trial.params.items():
        print(f"  {key}: {val}")
    best_parans = best_trial.params
    best_params.update({'probability': True, 'random_state': 42})
    return best_params



def evaluate_svm_pipeline(svm_pipeline, X_test, X_train, y_test, y_train):
    """
    Evaluates an SVM pipeline by computing ROC and Precision-Recall curves,
    selecting an optimal threshold via Youden's J statistic, and generating
    classification reports, confusion matrices, and probability boxplots.

    Parameters:
        svm_pipeline : Trained pipeline with a predict_proba method.
        X_test       : Features for the test set.
        X_train      : Features for the train set.
        y_test       : True labels for the test set.
        y_train      : True labels for the train set.
    """

    # ---------------------------
    # 1) Predict probabilities on the TEST set
    # ---------------------------
    y_pred_proba_test = svm_pipeline.predict_proba(X_test)[:, 1]  # probability for class=1

    # ---------------------------
    # 2) Compute ROC curve and AUC
    # ---------------------------
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba_test)
    roc_auc = roc_auc_score(y_test, y_pred_proba_test)
    print(f"Test ROC AUC: {roc_auc:.4f}")

    # ---------------------------
    # 3) Calculate Youden's J statistic
    # ---------------------------
    j_scores = tpr - fpr
    best_j_index = np.argmax(j_scores)
    best_threshold_j = thresholds[best_j_index]
    print(f"Best Threshold by Youden's J Statistic: {best_threshold_j:.4f}")

    # ---------------------------
    # 4) Plot ROC Curve
    # ---------------------------
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.2f})')
    plt.scatter(
        fpr[best_j_index],
        tpr[best_j_index],
        color='red',
        label=f'Threshold = {best_threshold_j:.4f}'
    )
    plt.plot([0, 1], [0, 1], '--', color='gray')
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('ROC Curve (Test)', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.show()

    # ---------------------------
    # 5) Apply the threshold to get class predictions on the TEST set
    # ---------------------------
    y_pred_j_test = (y_pred_proba_test >= best_threshold_j).astype(int)

    # ---------------------------
    # 6) Classification report & confusion matrix (TEST)
    # ---------------------------
    print("\nPerformance at Optimal Threshold (Youden's J) on Test Set:")
    print(classification_report(y_test, y_pred_j_test))

    conf_matrix_test = confusion_matrix(y_test, y_pred_j_test)
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix_test, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Pred Neg', 'Pred Pos'],
                yticklabels=['Actual Neg', 'Actual Pos'])
    plt.ylabel('Actual', fontsize=12)
    plt.xlabel('Predicted', fontsize=12)
    plt.title("Confusion Matrix (Test) @ Youden's J", fontsize=14)
    plt.show()

    # ---------------------------
    # 7) Directly compute precision, recall, and F1-score (TEST)
    # ---------------------------
    precision_test = precision_score(y_test, y_pred_j_test)
    recall_test = recall_score(y_test, y_pred_j_test)
    f1_test = f1_score(y_test, y_pred_j_test)

    print(f"Test Precision: {precision_test:.4f}")
    print(f"Test Recall:    {recall_test:.4f}")
    print(f"Test F1-Score:  {f1_test:.4f}")

    # ---------------------------
    # (Optional) Evaluate on TRAIN set with same threshold
    # ---------------------------
    y_pred_proba_train = svm_pipeline.predict_proba(X_train)[:, 1]  # probabilities on train
    y_pred_j_train = (y_pred_proba_train >= best_threshold_j).astype(int)  # apply same threshold

    print("\nPerformance at Optimal Threshold (Youden's J) on Training Set:")
    print(classification_report(y_train, y_pred_j_train))

    conf_matrix_train = confusion_matrix(y_train, y_pred_j_train)
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix_train, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Pred Neg', 'Pred Pos'],
                yticklabels=['Actual Neg', 'Actual Pos'])
    plt.ylabel('Actual', fontsize=12)
    plt.xlabel('Predicted', fontsize=12)
    plt.title("Confusion Matrix (Train) @ Youden's J", fontsize=14)
    plt.show()

    # ---------------------------
    # 8) Precision-Recall Curve Analysis (Test Set)
    # ---------------------------
    precision_vals, recall_vals, thresholds_pr = precision_recall_curve(y_test, y_pred_proba_test)
    avg_precision = average_precision_score(y_test, y_pred_proba_test)
    print(f"Test Average Precision (AP): {avg_precision:.4f}")

    plt.figure(figsize=(8, 6))
    plt.plot(recall_vals, precision_vals, label=f'Precision-Recall Curve (AP = {avg_precision:.2f})')
    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.title('Precision-Recall Curve (Test)', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.show()

    # Find the threshold on the precision-recall curve closest to our Youden threshold for reference
    if best_threshold_j < thresholds_pr[0]:
        pr_index = 0
    else:
        pr_index = np.argmin(np.abs(thresholds_pr - best_threshold_j))
    pr_at_youden = precision_vals[pr_index]
    recall_at_youden = recall_vals[pr_index]

    print(f"At Youden's J threshold ({best_threshold_j:.4f}):")
    print(f"  Precision ≈ {pr_at_youden:.4f}")
    print(f"  Recall    ≈ {recall_at_youden:.4f}")

    # ---------------------------
    # 9) Boxplot of Predicted Probabilities with Outcome Categories
    # ---------------------------
    # Create a dataframe for plotting
    df_plot = pd.DataFrame({
        'TrueLabel': y_test,
        'Predicted': y_pred_j_test,
        'Probability': y_pred_proba_test
    })

    # Map each row to an outcome category: TP, TN, FP, FN
    def get_outcome(row):
        if row['TrueLabel'] == 1 and row['Predicted'] == 1:
            return 'TP'
        elif row['TrueLabel'] == 0 and row['Predicted'] == 0:
            return 'TN'
        elif row['TrueLabel'] == 0 and row['Predicted'] == 1:
            return 'FP'
        else:
            return 'FN'

    df_plot['Outcome'] = df_plot.apply(get_outcome, axis=1)

    # Define the two-color palette
    palette = {
        False: "#00ADDC",  # Blue-ish color for negatives
        True:  "#F15A22"   # Orange-ish color for positives
    }

    # Mapping for outcome-specific styles (color, marker)
    outcome_styles = {
        'TP': {'color': palette[True],  'marker': 'o', 'label': 'TP'},
        'TN': {'color': palette[False], 'marker': 'o', 'label': 'TN'},
        'FP': {'color': palette[False], 'marker': 'x', 'label': 'FP'},
        'FN': {'color': palette[True],  'marker': 'x', 'label': 'FN'}
    }

    plt.figure(figsize=(5, 6))
    sns.set(style='whitegrid')

    # Draw the boxplot for Probability vs. Predicted class
    ax = sns.boxplot(
        x='Predicted',
        y='Probability',
        data=df_plot,
        palette=['white', 'white'],  # neutral white boxes
        showfliers=False,
        width=0.6
    )

    # Overlay points with outcome-specific markers and slight horizontal jitter
    for outcome, style_dict in outcome_styles.items():
        subset = df_plot[df_plot['Outcome'] == outcome]
        x_jitter = np.random.uniform(low=-0.05, high=0.05, size=len(subset))
        ax.scatter(
            subset['Predicted'] + x_jitter,
            subset['Probability'],
            color=style_dict['color'],
            marker=style_dict['marker'],
            edgecolor='black',
            alpha=0.7,
            label=style_dict['label']
        )

    ax.set_xlabel('Predicted Class', fontsize=12)
    ax.set_ylabel('Predicted Probability for Class=1', fontsize=12)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['0 (Pred Neg)', '1 (Pred Pos)'], fontsize=11)
    ax.set_title('Boxplot of Predicted Probability by Predicted Class\nWith Outcome-Specific Markers', fontsize=14)

    # Build a custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='white', label='TP',
               markerfacecolor=palette[True], markeredgecolor='black'),
        Line2D([0], [0], marker='o', color='white', label='TN',
               markerfacecolor=palette[False], markeredgecolor='black'),
        Line2D([0], [0], marker='x', color=palette[False], label='FP',
               markeredgecolor='black', markerfacecolor=palette[False]),
        Line2D([0], [0], marker='x', color=palette[True], label='FN',
               markeredgecolor='black', markerfacecolor=palette[True])
    ]
    ax.legend(
        handles=legend_elements,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.1),
        ncol=4
    )

    plt.tight_layout()
    plt.savefig('preds.svg', format='svg', bbox_inches='tight')
    plt.show()




def evaluate_shap(svm_pipeline, X_train, X_test, feature_names, save_path=None):
    """
    Evaluates and visualizes SHAP values for an SVM pipeline.
    
    Parameters:
        svm_pipeline  : Trained pipeline containing a 'scaler' and a 'svm' step.
        X_train       : Training features (unscaled).
        X_test        : Testing features (unscaled).
        feature_names : List of feature names.
        save_path     : Optional; if provided, the SHAP summary plot is saved as an SVG to this path.
        
    Returns:
        top_features  : Array of the top 20 features based on SHAP importance.
        feature_df    : DataFrame with feature names and their corresponding mean absolute SHAP values.
    """


    # 1) Extract the scaler and final svm model from the pipeline
    scaler = svm_pipeline.named_steps["scaler"]
    final_svm = svm_pipeline.named_steps["svm"]

    # 2) Scale the data
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # Optional: Create DataFrames for clarity
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=feature_names)
    X_test_scaled_df  = pd.DataFrame(X_test_scaled,  columns=feature_names)

    # 3) Create a SHAP explainer using the background training data
    explainer = shap.Explainer(
        final_svm.predict_proba,  # model function returning probabilities
        X_train_scaled_df,        # background data for reference
        feature_names=feature_names
    )
    
    # Compute SHAP values on the test set
    shap_values = explainer(X_test_scaled_df)
    shap_values_positive_class = shap_values[..., 1].values  # shape: (n_samples, n_features)
    
    # 4) Create a custom color map for the summary plot
    colors_list = ['#2F0C75', '#0063CA', '#0AA174', '#FFDD00']
    custom_cmap = mcolors.LinearSegmentedColormap.from_list('shap_custom_cmap', colors_list)

    # 5) Create the SHAP summary plot without automatically displaying it
    shap.summary_plot(
        shap_values_positive_class, 
        X_test_scaled_df, 
        feature_names=feature_names,
        show=True,  #  automatic display
        cmap=custom_cmap,
    )
    
    # If a save path is provided, save the figure as an SVG
    if save_path is not None:
        plt.savefig(save_path, format="svg", dpi=300, bbox_inches='tight')
    
    plt.show()

    # 6) Compute mean absolute SHAP values for each feature
    mean_abs_shap_values = np.abs(shap_values_positive_class).mean(axis=0)
    
    # Create a DataFrame with SHAP feature importance
    feature_df = pd.DataFrame({
        'feature_name': feature_names,
        'shap_importance': mean_abs_shap_values
    }).sort_values(by='shap_importance', ascending=False)

    # Print the top 20 features (you can change this number if needed)
    top_features = feature_df['feature_name'].head(20).values
    print("Top 20 Features (SHAP):", top_features)

    return top_features, feature_df.sort_values('shap_importance', ascending=False)

