import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import joblib

# File Paths
top_features_dataset_path = r'E:\bacillus_model\extracted_columns.csv'

# Load the dataset containing only the top features
top_features_df = pd.read_csv(top_features_dataset_path)
print(top_features_df.head())

# Separate Features and Target
X = top_features_df.drop(columns=['Label'])
y = top_features_df['Label']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Stratified K-Fold
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# SVM
param_grid_svm = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1e-3, 1e-4, 1e-5],
    'kernel': ['rbf']
}
grid_search_svm = GridSearchCV(SVC(random_state=42), param_grid_svm, cv=skf, scoring='accuracy', n_jobs=-1)
grid_search_svm.fit(X_train, y_train)
print(f"\nBest Parameters (SVM): {grid_search_svm.best_params_}")
print(f"Best Cross-Validation Score (SVM): {grid_search_svm.best_score_:.2f}")
best_svm_model = grid_search_svm.best_estimator_
y_svm_pred = best_svm_model.predict(X_test)
print("\nClassification Report (SVM Test Set):")
print(classification_report(y_test, y_svm_pred, zero_division=0))
cm_svm = confusion_matrix(y_test, y_svm_pred)
cmd_svm = ConfusionMatrixDisplay(cm_svm, display_labels=[0, 1])
cmd_svm.plot()
plt.title("Confusion Matrix (SVM)")
plt.show()

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_rf_pred = rf_model.predict(X_test)
print("\nClassification Report (Random Forest Test Set):")
print(classification_report(y_test, y_rf_pred, zero_division=0))
cm_rf = confusion_matrix(y_test, y_rf_pred)
cmd_rf = ConfusionMatrixDisplay(cm_rf, display_labels=[0, 1])
cmd_rf.plot()
plt.title("Confusion Matrix (Random Forest)")
plt.show()
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(10, 6))
plt.title("Random Forest Feature Importances")
plt.bar(range(X_train.shape[1]), importances[indices], align="center")
plt.xticks(range(X_train.shape[1]), [X.columns[i] for i in indices], rotation=90)
plt.tight_layout()
plt.show()

# XGBoost
param_grid_xgb = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}
grid_search_xgb = GridSearchCV(XGBClassifier(random_state=42, use_label_encoder=False), param_grid_xgb, cv=skf, scoring='accuracy', n_jobs=-1)
grid_search_xgb.fit(X_train, y_train)
print(f"\nBest Parameters (XGBoost): {grid_search_xgb.best_params_}")
print(f"Best Cross-Validation Score (XGBoost): {grid_search_xgb.best_score_:.2f}")
best_xgb_model = grid_search_xgb.best_estimator_
y_xgb_pred = best_xgb_model.predict(X_test)
print("\nClassification Report (XGBoost Test Set):")
print(classification_report(y_test, y_xgb_pred, zero_division=0))
cm_xgb = confusion_matrix(y_test, y_xgb_pred)
cmd_xgb = ConfusionMatrixDisplay(cm_xgb, display_labels=[0, 1])
cmd_xgb.plot()
plt.title("Confusion Matrix (XGBoost)")
plt.show()

# AdaBoost
param_grid_ada = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 1]
}
grid_search_ada = GridSearchCV(AdaBoostClassifier(random_state=42, algorithm='SAMME'), param_grid_ada, cv=skf, scoring='accuracy', n_jobs=-1)
grid_search_ada.fit(X_train, y_train)
print(f"\nBest Parameters (AdaBoost): {grid_search_ada.best_params_}")
print(f"Best Cross-Validation Score (AdaBoost): {grid_search_ada.best_score_:.2f}")
best_ada_model = grid_search_ada.best_estimator_
y_ada_pred = best_ada_model.predict(X_test)
print("\nClassification Report (AdaBoost Test Set):")
print(classification_report(y_test, y_ada_pred, zero_division=0))
cm_ada = confusion_matrix(y_test, y_ada_pred)
cmd_ada = ConfusionMatrixDisplay(cm_ada, display_labels=[0, 1])
cmd_ada.plot()
plt.title("Confusion Matrix (AdaBoost)")
plt.show()

# Naive Bayes
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
y_nb_pred = nb_model.predict(X_test)
print("\nClassification Report (Naive Bayes Test Set):")
print(classification_report(y_test, y_nb_pred, zero_division=0))
cm_nb = confusion_matrix(y_test, y_nb_pred)
cmd_nb = ConfusionMatrixDisplay(cm_nb, display_labels=[0, 1])
cmd_nb.plot()
plt.title("Confusion Matrix (Naive Bayes)")
plt.show()

# Save the models, selectors, and selected feature names
joblib.dump(best_svm_model, r'E:\bacillus_model\best_svm_model.pkl')
joblib.dump(rf_model, r'E:\bacillus_model\best_rf_model.pkl')
joblib.dump(best_xgb_model, r'E:\bacillus_model\best_xgb_model.pkl')
joblib.dump(best_ada_model, r'E:\bacillus_model\best_ada_model.pkl')
joblib.dump(nb_model, r'E:\bacillus_model\best_nb_model.pkl')
joblib.dump(X.columns.tolist(), r'E:\bacillus_model\selected_features.pkl')
