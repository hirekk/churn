"""Train random forest classifier for telecom customer churn prediction."""

import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import auc
from sklearn.metrics import average_precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

from churn.logger import DEFAULT_LOGGER
import mlflow


def train_random_forest(
    X: pd.DataFrame, y: pd.Series, random_state: int = 42
) -> RandomForestClassifier:
    """Train random forest with basic hyperparameter tuning."""
    logger = DEFAULT_LOGGER
    logger.info("Starting random forest training with hyperparameter tuning")

    # Basic hyperparameter grid
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [5, 10, 15, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }

    # Initialize base model
    base_rf = RandomForestClassifier(random_state=random_state, n_jobs=-1, class_weight="balanced")

    # Grid search with 5-fold CV - use PR-AUC for imbalanced dataset
    logger.info("Running grid search with 5-fold cross-validation using PR-AUC scoring")
    grid_search = GridSearchCV(
        estimator=base_rf,
        param_grid=param_grid,
        cv=5,
        scoring="average_precision",  # PR-AUC equivalent
        n_jobs=-1,
        verbose=1,
    )

    # Fit the grid search
    grid_search.fit(X, y)

    logger.info("Best parameters: %s", grid_search.best_params_)
    logger.info("Best CV PR-AUC score: %s", grid_search.best_score_)

    return grid_search.best_estimator_


def evaluate_model(model: RandomForestClassifier, X: pd.DataFrame, y: pd.Series) -> dict:
    """Evaluate model performance with cross-validation and detailed metrics."""
    logger = DEFAULT_LOGGER
    logger.info("Evaluating model performance")

    # Cross-validation scores - multiple metrics for imbalanced dataset
    cv_f1_scores = cross_val_score(model, X, y, cv=5, scoring="f1")
    cv_precision_scores = cross_val_score(model, X, y, cv=5, scoring="precision")
    cv_recall_scores = cross_val_score(model, X, y, cv=5, scoring="recall")
    cv_pr_auc_scores = cross_val_score(model, X, y, cv=5, scoring="average_precision")

    logger.info("5-fold CV F1 scores: %s", cv_f1_scores)
    logger.info("5-fold CV Precision scores: %s", cv_precision_scores)
    logger.info("5-fold CV Recall scores: %s", cv_recall_scores)
    logger.info("5-fold CV PR-AUC scores: %s", cv_pr_auc_scores)

    logger.info("Mean CV F1: %s (+/- %s)", cv_f1_scores.mean(), cv_f1_scores.std() * 2)
    logger.info(
        "Mean CV Precision: %s (+/- %s)", cv_precision_scores.mean(), cv_precision_scores.std() * 2
    )
    logger.info("Mean CV Recall: %s (+/- %s)", cv_recall_scores.mean(), cv_recall_scores.std() * 2)
    logger.info("Mean CV PR-AUC: %s (+/- %s)", cv_pr_auc_scores.mean(), cv_pr_auc_scores.std() * 2)

    # Predictions on full dataset
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)
    precision, recall, _ = precision_recall_curve(y, y_pred_proba[:, 1])
    pr_auc = auc(recall, precision)

    # Individual precision and recall scores
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)

    # PR-AUC score
    pr_auc = average_precision_score(y, y_pred_proba)

    logger.info("Full dataset metrics:")
    logger.info("  Precision: %s", precision)
    logger.info("  Recall: %s", recall)
    logger.info("  PR-AUC: %s", pr_auc)

    # Classification report
    report = classification_report(y, y_pred, output_dict=True)

    # Confusion matrix
    cm = confusion_matrix(y, y_pred)

    # Feature importance
    feature_importance = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    logger.info("Top 10 most important features:")
    for _, row in feature_importance.head(10).iterrows():
        logger.info("  %s: %s", row["feature"], row["importance"])

    # Save feature importance for MLflow artifact logging
    feature_importance.to_csv("feature_importance.csv", index=False)
    mlflow.log_artifact("feature_importance.csv")

    # Save confusion matrix for MLflow artifact logging
    cm_df = pd.DataFrame(
        data=cm, columns=["Predicted_0", "Predicted_1"], index=["Actual_0", "Actual_1"]
    )
    cm_df.to_csv("confusion_matrix.csv", index=True)
    mlflow.log_artifact("confusion_matrix.csv")

    return {
        "cv_f1_scores": cv_f1_scores,
        "cv_precision_scores": cv_precision_scores,
        "cv_recall_scores": cv_recall_scores,
        "cv_pr_auc_scores": cv_pr_auc_scores,
        "cv_f1_mean": cv_f1_scores.mean(),
        "cv_f1_std": cv_f1_scores.std(),
        "cv_precision_mean": cv_precision_scores.mean(),
        "cv_precision_std": cv_precision_scores.std(),
        "cv_recall_mean": cv_recall_scores.mean(),
        "cv_recall_std": cv_recall_scores.std(),
        "cv_pr_auc_mean": cv_pr_auc_scores.mean(),
        "cv_pr_auc_std": cv_pr_auc_scores.std(),
        "precision": precision,
        "recall": recall,
        "pr_auc": pr_auc,
        "classification_report": report,
        "confusion_matrix": cm,
        "feature_importance": feature_importance,
    }
