"""Train random forest classifier for telecom customer churn prediction."""

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import auc
from sklearn.metrics import average_precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from churn.data import CATEGORICAL_FEATURES
from churn.data import ORDINAL_FEATURES
from churn.logger import DEFAULT_LOGGER
import mlflow


def create_preprocessing_pipeline(
    categorical_features: list[str],
    ordinal_features: list[str],
    numerical_features: list[str],
) -> ColumnTransformer:
    """Create preprocessing pipeline with encoders and imputers.

    Args:
        categorical_features: List of categorical feature names
        ordinal_features: List of pre-binned ordinal feature names (from features.py)
        numerical_features: List of numerical feature names

    Returns:
        Preprocessing pipeline
    """
    logger = DEFAULT_LOGGER
    logger.info("Creating preprocessing pipeline")

    transformers = []

    if numerical_features:
        numerical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
            ],
        )
        transformers.append(("num", numerical_transformer, numerical_features))

    if categorical_features:
        categorical_transformer = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="constant", fill_value="missing"),
                ),
                (
                    "onehot",
                    OneHotEncoder(
                        drop="first",
                        sparse_output=False,
                        handle_unknown="ignore",
                    ),
                ),
            ],
        )
        transformers.append((
            "cat",
            categorical_transformer,
            categorical_features,
        ))

    if ordinal_features:
        ordinal_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
            ],
        )
        transformers.append(("ord", ordinal_transformer, ordinal_features))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def train_random_forest(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = 42,
    logger: logging.Logger = DEFAULT_LOGGER,
) -> Pipeline:
    """Train random forest with preprocessing pipeline and hyperparameter tuning.

    Args:
        X: Features dataframe
        y: Target variable series
        random_state: Random state for reproducibility
        logger: Logger instance

    Returns:
        Trained pipeline with best parameters
    """
    logger.info(
        "Starting random forest training with preprocessing pipeline and hyperparameter tuning",
    )

    all_features = set(X.columns)

    missing_categorical = set(CATEGORICAL_FEATURES) - all_features
    missing_ordinal = set(ORDINAL_FEATURES) - all_features

    if missing_categorical:
        logger.warning("Missing categorical features in dataset: %s", missing_categorical)
    if missing_ordinal:
        logger.warning("Missing ordinal features in dataset: %s", missing_ordinal)

    available_categorical = list(set(CATEGORICAL_FEATURES) & all_features)
    available_ordinal = list(set(ORDINAL_FEATURES) & all_features)
    numerical_features = list(all_features - set(available_ordinal) - set(available_categorical))

    logger.info("Categorical features: %s", available_categorical)
    logger.info("Ordinal features: %s", available_ordinal)
    logger.info("Numerical features: %s", numerical_features)

    preprocessor = create_preprocessing_pipeline(
        available_categorical,
        available_ordinal,
        numerical_features,
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(random_state=random_state, n_jobs=-1, class_weight="balanced"),
        ),
    ])

    param_grid = {
        "classifier__n_estimators": [20, 50, 100],
        "classifier__max_depth": [5, 10, 15],
        "classifier__min_samples_split": [5, 10, 50],
        "classifier__min_samples_leaf": [5, 10, 50],
    }

    logger.info("Running grid search with 3-fold cross-validation using PR-AUC scoring")
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=3,
        scoring="average_precision",
        n_jobs=-1,
        verbose=3,
    )

    grid_search.fit(X, y)

    logger.info("Best parameters: %s", grid_search.best_params_)
    logger.info("Best CV PR-AUC score: %s", grid_search.best_score_)

    return grid_search.best_estimator_


def evaluate_model(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    logger: logging.Logger = DEFAULT_LOGGER,
) -> dict[str, Any]:
    """Evaluate model performance with cross-validation and detailed metrics.

    Args:
        model: Trained model pipeline
        X: Features dataframe
        y: Target variable series
        logger: Logger instance

    Returns:
        Dictionary containing evaluation metrics
    """
    logger.info("Evaluating model performance")

    cv_f1_scores = cross_val_score(model, X, y, cv=5, scoring="f1")
    cv_precision_scores = cross_val_score(model, X, y, cv=5, scoring="precision")
    cv_recall_scores = cross_val_score(model, X, y, cv=5, scoring="recall")
    cv_pr_auc_scores = cross_val_score(model, X, y, cv=5, scoring="average_precision")

    logger.info("5-fold CV F1 scores: %s", cv_f1_scores)
    logger.info("5-fold CV Precision scores: %s", cv_precision_scores)
    logger.info("5-fold CV Recall scores: %s", cv_recall_scores)
    logger.info("5-fold CV PR-AUC scores: %s", cv_pr_auc_scores)

    logger.info("TRAINING CROSS-VALIDATION METRICS:")
    logger.info("  Mean CV F1: %s (+/- %s)", cv_f1_scores.mean(), cv_f1_scores.std() * 2)
    logger.info(
        "  Mean CV Precision: %s (+/- %s)",
        cv_precision_scores.mean(),
        cv_precision_scores.std() * 2,
    )
    logger.info(
        "  Mean CV Recall: %s (+/- %s)",
        cv_recall_scores.mean(),
        cv_recall_scores.std() * 2,
    )
    logger.info(
        "  Mean CV PR-AUC: %s (+/- %s)",
        cv_pr_auc_scores.mean(),
        cv_pr_auc_scores.std() * 2,
    )

    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)

    if y_pred_proba.ndim == 1:
        y_pred_proba = y_pred_proba.reshape(-1, 1)
        y_pred_proba = np.column_stack([1 - y_pred_proba, y_pred_proba])

    precision_curve, recall_curve, _ = precision_recall_curve(y, y_pred_proba[:, 1])
    pr_auc = auc(recall_curve, precision_curve)

    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)

    pr_auc_alt = average_precision_score(y, y_pred_proba[:, 1])

    logger.info("TRAINING DATASET METRICS (Full Dataset):")
    logger.info("  Precision: %s", precision)
    logger.info("  Recall: %s", recall)
    logger.info("  PR-AUC (curve): %s", pr_auc)
    logger.info("  PR-AUC (average_precision): %s", pr_auc_alt)

    report = classification_report(y, y_pred, output_dict=True)

    cm = confusion_matrix(y, y_pred)

    preprocessor = model.named_steps["preprocessor"]
    feature_names = preprocessor.get_feature_names_out()

    classifier = model.named_steps["classifier"]
    importances = classifier.feature_importances_

    feature_importance = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False)

    logger.info("Top 10 most important features:")
    for _, row in feature_importance.head(10).iterrows():
        logger.info("  %s: %s", row["feature"], row["importance"])

    feature_importance.to_csv("feature_importance.csv", index=False)
    mlflow.log_artifact("feature_importance.csv")

    cm_df = pd.DataFrame(
        data=cm,
        columns=pd.Index(["Predicted_0", "Predicted_1"]),
        index=pd.Index(["Actual_0", "Actual_1"]),
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
