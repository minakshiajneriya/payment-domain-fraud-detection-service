import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
import joblib

# Assuming these are already preprocessed
X_train = pd.read_csv('X_train_clean.csv')
y_train = pd.read_csv('y_train.csv').values.ravel()
X_test = pd.read_csv('X_test_clean.csv')
y_test = pd.read_csv('y_test.csv').values.ravel()

# Initialize base models
xgb_clf = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='logloss',
    random_state=42
)

rf_clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)

# Setup cross-validation
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros((X_train.shape[0], 2))
test_preds = np.zeros((X_test.shape[0], 2))

# Train base models and collect out-of-fold predictions
for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]

    xgb_clf.fit(X_tr, y_tr)
    rf_clf.fit(X_tr, y_tr)

    oof_preds[val_idx, 0] = xgb_clf.predict_proba(X_val)[:, 1]
    oof_preds[val_idx, 1] = rf_clf.predict_proba(X_val)[:, 1]

    test_preds[:, 0] += xgb_clf.predict_proba(X_test)[:, 1] / kf.n_splits
    test_preds[:, 1] += rf_clf.predict_proba(X_test)[:, 1] / kf.n_splits

# Train meta-model
meta_model = LogisticRegression()
meta_model.fit(oof_preds, y_train)

# Final predictions
y_pred = meta_model.predict_proba(test_preds)[:, 1]

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, meta_model.predict_proba(X_test)[:, 1])

joblib.dump(meta_model, "RandomForest_XGBoost.pkl")
print("Model saved successfully!")

print(f"\n{meta_model.__class__.__name__} Results:")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")
print(f"AUC-ROC:   {auc:.4f}")

# Evaluation
print("Meta model AUC on test set:", roc_auc_score(y_test, y_pred))
