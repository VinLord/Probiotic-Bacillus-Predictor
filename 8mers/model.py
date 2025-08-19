import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt
import joblib

# File Paths
features_scores_path = r'E:\bacillus_model\8mers\eightmerfeatures.csv'
main_dataset_path = r'E:\bacillus_model\8mers\eightmersdata.csv'

# Load Selected Features
features_scores_df = pd.read_csv(features_scores_path)
ranked_features = features_scores_df['Features'].tolist()

# Load the Main Dataset
main_df = pd.read_csv(main_dataset_path)

# Debug: Print the columns of the main dataset
print("Columns in main_df:", main_df.columns.tolist())

# Check if 'Label' column is present
if 'Label' not in main_df.columns:
    print("Error: 'Label' column not found in the dataset.")
else:
    # Separate Features and Target
    X = main_df.drop(columns=['Label'])
    y = main_df['Label']

    # Rank features using F-score (assuming already done in all5mersfeatures.csv)
    ranked_features_indices = [X.columns.get_loc(c) for c in ranked_features if c in X]

    # Initialize variables for IFS
    selected_features = []
    best_score = 0
    best_features = []
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    for feature_idx in ranked_features_indices:
        selected_features.append(feature_idx)
        model = SVC(random_state=42, kernel='linear')  # Use linear kernel for simplicity in IFS
        model.fit(X_train.iloc[:, selected_features], y_train)
        y_pred = model.predict(X_test.iloc[:, selected_features])
        score = accuracy_score(y_test, y_pred)

        if score > best_score:
            best_score = score
            best_features = selected_features.copy()
        else:
            selected_features.pop()

    # Transform indices to feature names
    selected_features_names = [X.columns[i] for i in best_features]

    # Display the selected features
    print("Selected Features:", selected_features_names)

    # Filter the Main Dataset to only include selected features
    main_filtered_df = main_df[selected_features_names + ['Label']]
    print(main_filtered_df.head())

    # Check the mean values of the selected features for each class
    feature_means = main_filtered_df.groupby('Label').mean()
    print("Mean values of selected features for each class:")
    print(feature_means)

    # Separate Features and Target with selected features
    X_selected = main_filtered_df.drop(columns=['Label'])
    y_selected = main_filtered_df['Label']

    # Cross-Validation
    model = SVC(random_state=42, kernel='linear')  # Use the same kernel as in IFS
    scores = cross_val_score(model, X_selected, y_selected, cv=10, scoring='accuracy')
    print(f"Cross-Validation Accuracy Scores: {scores}")
    print(f"Mean Accuracy: {scores.mean():.2f} ± {scores.std():.2f}")

    # Split again for final training and testing
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y_selected, test_size=0.2, random_state=42)

    # Define the parameter grid for SVM
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': [1e-3, 1e-4, 1e-5],
        'kernel': ['rbf']
    }

    # Initialize the Stratified K-Fold
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # Initialize the Grid Search for SVM
    grid_search = GridSearchCV(SVC(random_state=42), param_grid, cv=skf, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Print the best parameters and best score
    print(f"\nBest Parameters (SVM): {grid_search.best_params_}")
    print(f"Best Cross-Validation Score (SVM): {grid_search.best_score_:.2f}")

    # Use the best estimator for testing predictions
    best_svm_model = grid_search.best_estimator_

    y_svm_pred = best_svm_model.predict(X_test)

    # Print the Classification Report for SVM
    print("\nClassification Report (SVM Test Set):")
    print(classification_report(y_test, y_svm_pred, zero_division=0))

    # Confusion Matrix Display for SVM
    cm_svm = confusion_matrix(y_test, y_svm_pred)
    cmd_svm = ConfusionMatrixDisplay(cm_svm, display_labels=[0, 1])
    cmd_svm.plot()
    plt.title("Confusion Matrix (SVM)")
    plt.show()

    # Initialize and Train the Random Forest Model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # Evaluate the Random Forest Model
    y_rf_pred = rf_model.predict(X_test)
    print("\nClassification Report (Random Forest Test Set):")
    print(classification_report(y_test, y_rf_pred, zero_division=0))

    # Confusion Matrix Display for Random Forest
    cm_rf = confusion_matrix(y_test, y_rf_pred)
    cmd_rf = ConfusionMatrixDisplay(cm_rf, display_labels=[0, 1])
    cmd_rf.plot()
    plt.title("Confusion Matrix (Random Forest)")
    plt.show()

    # Plot Feature Importances for Random Forest
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.title("Random Forest Feature Importances")
    plt.bar(range(X_train.shape[1]), importances[indices], align="center")
    plt.xticks(range(X_train.shape[1]), [X_selected.columns[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.show()

    # Save the models, selectors, and selected feature names
    joblib.dump(best_svm_model, r'E:\bacillus_model\8mers\best_svm_model.pkl')
    joblib.dump(rf_model, r'E:\bacillus_model\8mers\best_rf_model.pkl')
    joblib.dump(selected_features_names, r'E:\bacillus_model\8mers\selected_features.pkl')
