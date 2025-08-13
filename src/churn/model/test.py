"""Test trained model on test dataset for reliable performance evaluation."""

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import auc
from sklearn.metrics import average_precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.pipeline import Pipeline

from churn.logger import DEFAULT_LOGGER
import mlflow


def evaluate_test_performance(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_path: Path,
    logger: logging.Logger = DEFAULT_LOGGER,
) -> dict[str, Any]:
    """Evaluate model performance on test dataset.

    Args:
        model: Trained model pipeline
        X_test: Test features
        y_test: Test target variable
        model_path: Path to model directory
        logger: Logger instance

    Returns:
        Dictionary containing test performance metrics
    """
    logger.info("Evaluating TEST dataset performance")
    logger.info("=" * 50)

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba[:, 1])
    pr_auc = auc(recall_curve, precision_curve)

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    pr_auc_avg = average_precision_score(y_test, y_pred_proba[:, 1])

    f1 = f1_score(y_test, y_pred)

    logger.info("TEST DATASET METRICS:")
    logger.info("  F1 Score: %s", f1)
    logger.info("  Precision: %s", precision)
    logger.info("  Recall: %s", recall)
    logger.info("  PR-AUC (curve): %s", pr_auc)
    logger.info("  PR-AUC (average_precision): %s", pr_auc_avg)
    logger.info("=" * 50)

    report = classification_report(y_test, y_pred, output_dict=True)

    cm = confusion_matrix(y_test, y_pred)

    preprocessor = model.named_steps["preprocessor"]
    feature_names = preprocessor.get_feature_names_out()

    classifier = model.named_steps["classifier"]
    importances = classifier.feature_importances_

    feature_importance = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False)

    logger.info("Top 10 most important features (from test evaluation):")
    for _, row in feature_importance.head(10).iterrows():
        logger.info("  %s: %s", row["feature"], row["importance"])

    test_results = {
        "test_f1": f1,
        "test_precision": precision,
        "test_recall": recall,
        "test_pr_auc_curve": pr_auc,
        "test_pr_auc_avg": pr_auc_avg,
        "test_classification_report": report,
        "test_confusion_matrix": cm.tolist(),
        "test_feature_importance": feature_importance.to_dict("records"),
        "test_samples": len(X_test),
        "test_features": len(X_test.columns),
        "test_target_distribution": y_test.value_counts().to_dict(),
    }

    test_results_path = model_path / "test_results.json"
    with test_results_path.open(mode="w", encoding="utf-8") as f:
        json.dump(test_results, f, indent=2, default=str)

    logger.info("Test results saved to %s", test_results_path)

    feature_importance.to_csv("test_feature_importance.csv", index=False)
    mlflow.log_artifact("test_feature_importance.csv")

    cm_df = pd.DataFrame(
        data=cm,
        columns=pd.Index(["Predicted_0", "Predicted_1"]),
        index=pd.Index(["Actual_0", "Actual_1"]),
    )
    cm_df.to_csv("test_confusion_matrix.csv", index=True)
    mlflow.log_artifact("test_confusion_matrix.csv")

    mlflow.log_metric("test_f1", f1)
    mlflow.log_metric("test_precision", precision)
    mlflow.log_metric("test_recall", recall)
    mlflow.log_metric("test_pr_auc", pr_auc)
    mlflow.log_metric("test_samples", len(X_test))
    mlflow.log_metric("test_features", len(X_test.columns))

    return test_results
