"""
Typing classification model.
Class labels: 0 = Healthy Control, 1 = Cognitive Decline
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.evaluation.metrics import calculate_metrics


def train_typing_model(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(
            n_estimators=150, max_depth=6, class_weight="balanced", random_state=random_state
        ))
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_f1 = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    results = calculate_metrics(y_test, y_pred, y_prob)
    results["cv_f1_mean"] = float(cv_f1.mean())
    results["cv_f1_std"] = float(cv_f1.std())
    results["y_test"] = y_test
    results["y_pred"] = y_pred

    return model, results