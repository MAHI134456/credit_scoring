import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# ======================
# Load data
# ======================
df = pd.read_csv('data/processed/model_ready_with_target.csv')

# Split features & target
X = df.drop(columns=['CustomerId', 'is_high_risk'])
y = df['is_high_risk']

# ======================
# Train-test split
# ======================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ======================
# Setup MLflow experiment
# ======================
mlflow.set_experiment("credit_scoring")

# ======================
# Models & Hyperparameters
# ======================
models = {
    'logistic_regression': {
        'model': LogisticRegression(max_iter=1000),
        'param_grid': {
            'C': [0.1, 1.0, 10.0],
            'solver': ['liblinear']
        }
    },
    'random_forest': {
        'model': RandomForestClassifier(),
        'param_grid': {
            'n_estimators': [50, 100],
            'max_depth': [5, 10]
        }
    }
}

# ======================
# Train & log experiments
# ======================
for name, config in models.items():
    with mlflow.start_run(run_name=name):
        clf = GridSearchCV(
            estimator=config['model'],
            param_grid=config['param_grid'],
            cv=3,
            scoring='roc_auc',
            n_jobs=-1
        )
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_proba)

        mlflow.log_params(clf.best_params_)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc)

        # Save model
        mlflow.sklearn.log_model(clf.best_estimator_, "model")

        print(f"✅ {name} | Best params: {clf.best_params_} | AUC: {roc:.3f}")

print("✅ Training & tracking done.")
