from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import joblib
import matplotlib.pyplot as plt

X_test = pd.read_csv("X_test_clean.csv")
y_test = pd.read_csv("y_test.csv")

model = joblib.load("CatBoost_model.pkl")
y_pred = model.predict(X_test)  #model will predict the result for X_test 
cm = confusion_matrix(y_test, y_pred) # comparison of actual vs real

# Plot confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix - Credit Card Fraud Detection with CatBoost")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()