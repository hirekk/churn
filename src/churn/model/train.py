"""Train random forest classifier for telecom customer churn prediction."""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
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

    # Grid search with 5-fold CV
    logger.info("Running grid search with 5-fold cross-validation")
    grid_search = GridSearchCV(
        estimator=base_rf, param_grid=param_grid, cv=5, scoring="f1", n_jobs=-1, verbose=1
    )

    # Fit the grid search
    grid_search.fit(X, y)

    logger.info("Best parameters: %s", grid_search.best_params_)
    logger.info("Best CV score: %s", grid_search.best_score_)

    # Log best parameters to MLflow
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric("best_cv_score", grid_search.best_score_)

    return grid_search.best_estimator_


def evaluate_model(model: RandomForestClassifier, X: pd.DataFrame, y: pd.Series) -> dict:
    """Evaluate model performance with cross-validation and detailed metrics."""
    logger = DEFAULT_LOGGER
    logger.info("Evaluating model performance")

    # Cross-validation scores
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="f1")
    logger.info("5-fold CV F1 scores: %s", cv_scores)
    logger.info("Mean CV F1: %s (+/- %s)", cv_scores.mean(), cv_scores.std() * 2)

    # Log CV metrics to MLflow
    mlflow.log_metric("cv_f1_mean", cv_scores.mean())
    mlflow.log_metric("cv_f1_std", cv_scores.std())
    mlflow.log_metric("cv_f1_min", cv_scores.min())
    mlflow.log_metric("cv_f1_max", cv_scores.max())

    # Predictions on full dataset
    y_pred = model.predict(X)

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

    # Log feature importance to MLflow
    feature_importance.to_csv("feature_importance.csv", index=False)
    mlflow.log_artifact("feature_importance.csv")

    # Log confusion matrix to MLflow
    cm_df = pd.DataFrame(cm, columns=["Predicted_0", "Predicted_1"], index=["Actual_0", "Actual_1"])
    cm_df.to_csv("confusion_matrix.csv", index=True)
    mlflow.log_artifact("confusion_matrix.csv")

    return {
        "cv_scores": cv_scores,
        "cv_mean": cv_scores.mean(),
        "cv_std": cv_scores.std(),
        "classification_report": report,
        "confusion_matrix": cm,
        "feature_importance": feature_importance,
    }
