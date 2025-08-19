import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt
import joblib
from xgboost import XGBClassifier

# File Paths
features_scores_path = r'E:\bacillus_model\6mers\6merfeatures.csv'
main_dataset_path = r'E:\bacillus_model\6mers\sixmersdata.csv'

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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Check the distribution of labels
    print("Label distribution in the entire dataset:")
    print(y.value_counts())
    print("Label distribution in the training dataset:")
    print(y_train.value_counts())
    print("Label distribution in the testing dataset:")
    print(y_test.value_counts())

    # Select top N features (e.g., 5 features)
    top_n = 10
    for feature_idx in ranked_features_indices[:top_n]:
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
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y_selected, test_size=0.1, random_state=42, stratify=y_selected)

    # Define the parameter grid for each model
    param_grid_svm = {
        'C': [0.1, 1, 10, 100],
        'gamma': [1e-3, 1e-4, 1e-5],
        'kernel': ['rbf']
    }

    param_grid_rf = {
        'n_estimators': [50, 100, 200],
        'max_features': ['sqrt', 'log2', None],
        'max_depth': [4, 6, 8],
        'criterion': ['gini', 'entropy']
    }

    param_grid_xgb = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 4, 5]
    }

    param_grid_nb = {}  # Naive Bayes doesn't have hyperparameters to tune

    param_grid_ada = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 1]
    }

    # Initialize the Stratified K-Fold
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # Grid Search for each model
    def grid_search(model, param_grid, X_train, y_train):
        grid_search = GridSearchCV(model, param_grid, cv=skf, scoring='accuracy', n_jobs=-1, error_score='raise')
        grid_search.fit(X_train, y_train)
        return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_

    # SVM
    best_svm_model, best_svm_params, best_svm_score = grid_search(SVC(random_state=42), param_grid_svm, X_train, y_train)
    print(f"\nBest Parameters (SVM): {best_svm_params}")
    print(f"Best Cross-Validation Score (SVM): {best_svm_score:.2f}")

    # Random Forest
    best_rf_model, best_rf_params, best_rf_score = grid_search(RandomForestClassifier(random_state=42), param_grid_rf, X_train, y_train)
    print(f"\nBest Parameters (Random Forest): {best_rf_params}")
    print(f"Best Cross-Validation Score (Random Forest): {best_rf_score:.2f}")

    # XGBoost
    best_xgb_model, best_xgb_params, best_xgb_score = grid_search(XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'), param_grid_xgb, X_train, y_train)
    print(f"\nBest Parameters (XGBoost): {best_xgb_params}")
    print(f"Best Cross-Validation Score (XGBoost): {best_xgb_score:.2f}")

    # Naive Bayes
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)
    nb_score = cross_val_score(nb_model, X_train, y_train, cv=skf, scoring='accuracy').mean()
    print(f"\nCross-Validation Score (Naive Bayes): {nb_score:.2f}")

    # AdaBoost
    best_ada_model, best_ada_params, best_ada_score = grid_search(AdaBoostClassifier(random_state=42, algorithm='SAMME'), param_grid_ada, X_train, y_train)
    print(f"\nBest Parameters (AdaBoost): {best_ada_params}")
    print(f"Best Cross-Validation Score (AdaBoost): {best_ada_score:.2f}")

    # Evaluation and Plotting
    models = {
        'SVM': best_svm_model,
        'Random Forest': best_rf_model,
        'XGBoost': best_xgb_model,
        'Naive Bayes': nb_model,
        'AdaBoost': best_ada_model
    }

    for model_name, model in models.items():
        print(f"\nEvaluating {model_name}...")
        y_pred = model.predict(X_test)
        print(f"\nClassification Report ({model_name} Test Set):")
        print(classification_report(y_test, y_pred, zero_division=0))

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        cmd = ConfusionMatrixDisplay(cm, display_labels=[0, 1])
        cmd.plot()
        plt.title(f"Confusion Matrix ({model_name})")
        plt.show()

        # Feature Importances (if applicable)
        if model_name in ['Random Forest', 'XGBoost']:
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]

            plt.figure(figsize=(10, 6))
            plt.title(f"{model_name} Feature Importances")
            plt.bar(range(X_train.shape[1]), importances[indices], align="center")
            plt.xticks(range(X_train.shape[1]), [X_selected.columns[i] for i in indices], rotation=90)
            plt.tight_layout()
            plt.show()


    # Save the models and selected feature names
    joblib.dump(best_svm_model, r'E:\bacillus_model\6mers\best_svm_model.pkl')
    joblib.dump(best_rf_model, r'E:\bacillus_model\6mers\best_rf_model.pkl')
    joblib.dump(best_xgb_model, r'E:\bacillus_model\6mers\best_xgb_model.pkl')
    joblib.dump(nb_model, r'E:\bacillus_model\6mers\best_nb_model.pkl')
    joblib.dump(best_ada_model, r'E:\bacillus_model\6mers\best_ada_model.pkl')
    joblib.dump(selected_features_names, r'E:\bacillus_model\6mers\selected_features.pkl')
