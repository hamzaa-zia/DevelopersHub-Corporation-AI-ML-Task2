"""
Heart Disease EDA and Classification Model

This script loads the heart disease dataset, performs basic cleaning checks,
creates exploratory visualizations, trains classification models, evaluates
them, and saves the most important outputs for reporting.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


DATA_PATH = Path("heart.csv")
OUTPUT_DIR = Path("outputs")
RANDOM_STATE = 42

CATEGORICAL_COLUMNS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
NUMERIC_COLUMNS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
TARGET_COLUMN = "target"

FEATURE_LABELS = {
    "age": "Age",
    "sex": "Sex",
    "cp": "Chest pain type",
    "trestbps": "Resting blood pressure",
    "chol": "Serum cholesterol",
    "fbs": "Fasting blood sugar > 120 mg/dL",
    "restecg": "Resting ECG result",
    "thalach": "Maximum heart rate achieved",
    "exang": "Exercise-induced angina",
    "oldpeak": "Exercise ST depression",
    "slope": "Peak exercise ST slope",
    "ca": "Number of major vessels",
    "thal": "Thalassemia test result",
}

CATEGORY_VALUE_NOTES = {
    "sex": "0 = Female, 1 = Male",
    "cp": "0-3 = Chest pain category codes",
    "fbs": "0 = <= 120 mg/dL, 1 = > 120 mg/dL",
    "restecg": "0-2 = Resting ECG result codes",
    "exang": "0 = No, 1 = Yes",
    "slope": "0-2 = Exercise ST slope category codes",
    "ca": "0-4 = Major vessels colored by fluoroscopy",
    "thal": "1-3 = Thalassemia test category codes",
}


def load_dataset(path: Path) -> pd.DataFrame:
    """Load the CSV dataset and return it as a pandas DataFrame."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path)


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset for modeling.

    The current dataset has no missing values, but the function still handles
    them so the workflow remains complete and reproducible.
    """
    cleaned = df.copy()

    for column in NUMERIC_COLUMNS:
        cleaned[column] = cleaned[column].fillna(cleaned[column].median())

    for column in CATEGORICAL_COLUMNS + [TARGET_COLUMN]:
        cleaned[column] = cleaned[column].fillna(cleaned[column].mode()[0])

    return cleaned


def save_dataset_summary(df: pd.DataFrame, cleaned_df: pd.DataFrame) -> None:
    """Save basic dataset details, missing values, and descriptive statistics."""
    summary_path = OUTPUT_DIR / "dataset_summary.txt"

    with summary_path.open("w", encoding="utf-8") as file:
        file.write("Heart Disease Dataset Summary\n")
        file.write("=============================\n\n")
        file.write(f"Original shape: {df.shape[0]} rows, {df.shape[1]} columns\n")
        file.write(f"Cleaned shape: {cleaned_df.shape[0]} rows, {cleaned_df.shape[1]} columns\n\n")
        file.write("Columns:\n")
        for column in cleaned_df.columns:
            file.write(f"- {column}: {cleaned_df[column].dtype}\n")

        file.write("\nMissing values before cleaning:\n")
        file.write(df.isna().sum().to_string())
        file.write("\n\nTarget distribution:\n")
        file.write(cleaned_df[TARGET_COLUMN].value_counts().sort_index().to_string())
        file.write("\n\nDescriptive statistics:\n")
        file.write(cleaned_df.describe().round(2).to_string())


def create_eda_plots(df: pd.DataFrame) -> None:
    """Create EDA charts that show target balance and feature trends."""
    sns.set_theme(style="whitegrid")

    target_labels = {0: "No heart disease", 1: "Heart disease"}

    plt.figure(figsize=(7, 5))
    ax = sns.countplot(data=df, x=TARGET_COLUMN, hue=TARGET_COLUMN, palette="Set2", legend=False)
    ax.set_title("Heart Disease Target Distribution")
    ax.set_xlabel("Diagnosis group (0 = No heart disease, 1 = Heart disease)")
    ax.set_ylabel("Number of patients")
    ax.set_xticks([0, 1])
    ax.set_xticklabels([target_labels[0], target_labels[1]])
    for container in ax.containers:
        ax.bar_label(container, fmt="%d")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "target_distribution.png", dpi=160)
    plt.close()

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, column in zip(axes.flat, NUMERIC_COLUMNS):
        sns.histplot(data=df, x=column, hue=TARGET_COLUMN, kde=True, bins=20, ax=ax, palette="Set1")
        ax.set_title(f"{FEATURE_LABELS[column]} by Diagnosis")
        ax.set_xlabel(FEATURE_LABELS[column])
        ax.set_ylabel("Patient count")
        legend = ax.get_legend()
        if legend:
            legend.set_title("Target")
            for text, label in zip(legend.texts, ["No disease (0)", "Disease (1)"]):
                text.set_text(label)
    axes.flat[-1].axis("off")
    fig.text(
        0.5,
        0.02,
        "Bar height shows patient count in each value range. Curved lines show the distribution trend.",
        ha="center",
    )
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)
    plt.savefig(OUTPUT_DIR / "numeric_feature_distributions.png", dpi=160)
    plt.close()

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    for ax, column in zip(axes.flat, CATEGORICAL_COLUMNS):
        sns.countplot(data=df, x=column, hue=TARGET_COLUMN, ax=ax, palette="Set2")
        ax.set_title(f"{FEATURE_LABELS[column]} vs Diagnosis")
        ax.set_xlabel(FEATURE_LABELS[column])
        ax.set_ylabel("Patient count")
        ax.legend(title="Target", labels=["No disease", "Disease"])
        ax.text(
            0.5,
            -0.28,
            CATEGORY_VALUE_NOTES[column],
            ha="center",
            va="top",
            transform=ax.transAxes,
            fontsize=8,
            wrap=True,
        )
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.55)
    plt.savefig(OUTPUT_DIR / "categorical_feature_trends.png", dpi=160)
    plt.close()

    plt.figure(figsize=(12, 9))
    correlation = df.corr(numeric_only=True)
    sns.heatmap(correlation, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.4)
    plt.title("Feature Correlation Heatmap")
    plt.figtext(
        0.5,
        0.01,
        "Values range from -1 to +1. Positive values move together, negative values move opposite, and values near 0 show weak linear relation.",
        ha="center",
        fontsize=9,
    )
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)
    plt.savefig(OUTPUT_DIR / "correlation_heatmap.png", dpi=160)
    plt.close()


def build_preprocessor() -> ColumnTransformer:
    """Build preprocessing for numeric scaling and categorical one-hot encoding."""
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_COLUMNS),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS),
        ]
    )


def train_models(X_train: pd.DataFrame, y_train: pd.Series) -> dict[str, Pipeline]:
    """Train Logistic Regression and Decision Tree classification models."""
    models = {
        "Logistic Regression": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
            ]
        ),
        "Decision Tree": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("model", DecisionTreeClassifier(max_depth=4, random_state=RANDOM_STATE)),
            ]
        ),
    }

    for model in models.values():
        model.fit(X_train, y_train)

    return models


def get_feature_names(model: Pipeline) -> list[str]:
    """Return readable feature names after preprocessing."""
    preprocessor = model.named_steps["preprocessor"]
    encoded_names = preprocessor.named_transformers_["categorical"].get_feature_names_out(CATEGORICAL_COLUMNS)
    return NUMERIC_COLUMNS + encoded_names.tolist()


def evaluate_models(models: dict[str, Pipeline], X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    """Evaluate models and save confusion matrices, ROC curves, and reports."""
    rows = []
    plt.figure(figsize=(8, 6))

    for model_name, model in models.items():
        y_pred = model.predict(X_test)
        y_probability = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_probability)
        rows.append({"model": model_name, "accuracy": accuracy, "roc_auc": auc_score})

        report_path = OUTPUT_DIR / f"{model_name.lower().replace(' ', '_')}_classification_report.txt"
        with report_path.open("w", encoding="utf-8") as file:
            file.write(f"{model_name} Classification Report\n")
            file.write("=" * (len(model_name) + 22))
            file.write("\n\n")
            file.write(classification_report(y_test, y_pred, target_names=["No disease", "Disease"]))
            file.write("\nConfusion matrix:\n")
            file.write(str(confusion_matrix(y_test, y_pred)))

        ConfusionMatrixDisplay.from_predictions(
            y_test,
            y_pred,
            display_labels=["No disease", "Disease"],
            cmap="Blues",
            colorbar=False,
        )
        plt.title(f"{model_name} Confusion Matrix")
        plt.xlabel("Predicted label")
        plt.ylabel("Actual label")
        plt.figtext(
            0.5,
            0.01,
            "Diagonal cells are correct predictions. Off-diagonal cells are model errors.",
            ha="center",
            fontsize=9,
        )
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.14)
        plt.savefig(OUTPUT_DIR / f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png", dpi=160)
        plt.close()

        false_positive_rate, true_positive_rate, _ = roc_curve(y_test, y_probability)
        plt.plot(false_positive_rate, true_positive_rate, label=f"{model_name} (AUC = {auc_score:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
    plt.title("ROC Curve Comparison")
    plt.xlabel("False Positive Rate (patients incorrectly predicted as disease)")
    plt.ylabel("True Positive Rate / Recall (disease patients correctly found)")
    plt.legend()
    plt.figtext(
        0.5,
        0.01,
        "AUC closer to 1.0 means stronger class separation. The dashed line represents random guessing.",
        ha="center",
        fontsize=9,
    )
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.14)
    plt.savefig(OUTPUT_DIR / "roc_curve_comparison.png", dpi=160)
    plt.close()

    metrics = pd.DataFrame(rows).sort_values(by="roc_auc", ascending=False)
    metrics.to_csv(OUTPUT_DIR / "model_metrics.csv", index=False)
    return metrics


def save_feature_importance(models: dict[str, Pipeline]) -> None:
    """Save and plot feature importance for both trained models."""
    for model_name, model in models.items():
        feature_names = get_feature_names(model)
        estimator = model.named_steps["model"]

        if model_name == "Logistic Regression":
            values = abs(estimator.coef_[0])
            importance_label = "Absolute coefficient"
        else:
            values = estimator.feature_importances_
            importance_label = "Tree importance"

        importance = (
            pd.DataFrame({"feature": feature_names, "importance": values})
            .sort_values(by="importance", ascending=False)
            .head(10)
        )
        importance.to_csv(OUTPUT_DIR / f"{model_name.lower().replace(' ', '_')}_feature_importance.csv", index=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(data=importance, y="feature", x="importance", hue="feature", palette="viridis", legend=False)
        plt.title(f"Top Features - {model_name}")
        plt.xlabel(importance_label)
        plt.ylabel("Feature")
        plt.figtext(
            0.5,
            0.01,
            "Longer bars mean the feature had more influence in this model. Encoded names like cp_0 represent category value 0 for that feature.",
            ha="center",
            fontsize=9,
        )
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.14)
        plt.savefig(OUTPUT_DIR / f"{model_name.lower().replace(' ', '_')}_feature_importance.png", dpi=160)
        plt.close()


def main() -> None:
    """Run the complete EDA, modeling, and evaluation workflow."""
    OUTPUT_DIR.mkdir(exist_ok=True)

    df = load_dataset(DATA_PATH)
    cleaned_df = clean_dataset(df)
    save_dataset_summary(df, cleaned_df)
    create_eda_plots(cleaned_df)

    X = cleaned_df.drop(columns=[TARGET_COLUMN])
    y = cleaned_df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    models = train_models(X_train, y_train)
    metrics = evaluate_models(models, X_test, y_test)
    save_feature_importance(models)

    print("Heart disease EDA and modeling workflow completed.")
    print(f"Rows analyzed: {cleaned_df.shape[0]}")
    print("\nModel metrics:")
    print(metrics.round(3).to_string(index=False))
    print(f"\nOutputs saved in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
