"""
CoolGuard - Train a real personalization model (genuine AI, not just an average)

This replaces the "average of past safe readings" trick with an actual small
machine learning model: logistic regression predicting the probability that a
given combination of WBGT, heart-rate-rise, age, and moisture leads to a
REST/STOP outcome, learned from YOUR OWN logged data.

Why logistic regression, and why say so honestly: it's simple enough to
explain fully in a judge interview (a weighted sum of your inputs, squashed
into a 0-1 probability), and with a small dataset (which is what a school
project realistically collects) a simple model generalizes better and is
less likely to overfit than a complex one. Say this in your report, "we chose
a simple, explainable model appropriate for our dataset size" is a genuinely
strong thing to say to a judge, not a weakness.

Usage:
  1. Collect data first: use the CoolGuard app for real (or simulated) checks
     until coolguard_log.csv (downloaded from the Worker log tab) has at
     least ~30-50 rows. More is better.
  2. pip install scikit-learn pandas joblib
  3. python train_personalization_model.py coolguard_log.csv
  4. This produces coolguard_model.joblib, which the app can load.
"""

import sys

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

FEATURES = ["wbgt", "current_hr", "resting_hr", "age", "moisture"]


def load_and_prepare(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Log is missing expected columns: {missing}")
    df["hr_rise"] = df["current_hr"] - df["resting_hr"]
    df["needs_rest"] = df["decision"].isin(["🔴 STOP", "🟡 REST SOON"]).astype(int)
    return df


def train(df: pd.DataFrame):
    X = df[["wbgt", "hr_rise", "age", "moisture"]]
    y = df["needs_rest"]

    if y.nunique() < 2:
        raise ValueError(
            "Your log only contains one outcome type so far (all safe, or all "
            "rest/stop). Collect a wider range of readings, some clearly safe "
            "and some clearly risky, before training."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    model = LogisticRegression()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print("Test accuracy:", round(accuracy_score(y_test, preds), 3))
    print(classification_report(y_test, preds, zero_division=0))
    print("\nLearned weights (how much each factor pushes toward 'needs rest'):")
    for name, coef in zip(X.columns, model.coef_[0]):
        print(f"  {name:>10}: {coef:+.3f}")

    return model


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python train_personalization_model.py <path_to_log.csv>")
        sys.exit(1)

    data = load_and_prepare(sys.argv[1])
    print(f"Loaded {len(data)} logged readings.")
    if len(data) < 20:
        print("Warning: fewer than 20 rows. The model will train, but with this "
              "little data the coefficients won't be very reliable yet. Say so "
              "honestly in your report, and keep collecting data.")

    trained_model = train(data)
    joblib.dump(trained_model, "coolguard_model.joblib")
    print("\nSaved coolguard_model.joblib. Place it next to coolguard_app.py.")
