import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import re
# ✅ Function to evaluate model using CSV input
def evaluate_model_from_csv(model, X_train_file, y_train_file, X_test_file, y_test_file):
    import pandas as pd
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    # Load the CSV files
    X_train = pd.read_csv(X_train_file)
    y_train = pd.read_csv(y_train_file) # Ensure y is a 1D array
    X_test = pd.read_csv(X_test_file)
    y_test = pd.read_csv(y_test_file)

    # Convert to 1D array (if necessary)
    if len(y_train.shape) > 1:
        y_train = y_train.values.ravel()

    if len(y_test.shape) > 1:
        y_test = y_test.values.ravel()

    # Fit the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    joblib.dump(model, "CatBoost_model.pkl")
    print("Model saved successfully!")

    print(f"\n{model.__class__.__name__} Results:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"AUC-ROC:   {auc:.4f}")



# ✅ Paths to the CSV files
X_train_file = "X_train_clean.csv"
y_train_file = "y_train.csv"
X_test_file = "X_test_clean.csv"
y_test_file = "y_test.csv"

# ✅ Models to train    
models = {
    #"Logistic Regression": LogisticRegression(max_iter=1000)
    #"Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    #"XGBoost": XGBClassifier(n_estimators=100, random_state=42)
   # "LightGBM": LGBMClassifier(n_estimators=100, random_state=42),
    "CatBoost": CatBoostClassifier(verbose=0, random_state=42)
}

# ✅ Train and evaluate each model using CSV inputs
for name, model in models.items():
    print(f"\nTraining {name} with CSV input...")
    evaluate_model_from_csv(model, X_train_file, y_train_file, X_test_file, y_test_file)
