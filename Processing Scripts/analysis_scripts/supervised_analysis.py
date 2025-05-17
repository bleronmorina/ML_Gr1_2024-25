import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

def main():
    potential_target_cols = [
        'DALYs (rate) - Sex: Both - Age: Age-standardized - Cause: Depressive disorders',
        'DALYs (rate) - Sex: Both - Age: Age-standardized - Cause: Schizophrenia',
        'DALYs (rate) - Sex: Both - Age: Age-standardized - Cause: Bipolar disorder',
        'DALYs (rate) - Sex: Both - Age: Age-standardized - Cause: Eating disorders',
        'DALYs (rate) - Sex: Both - Age: Age-standardized - Cause: Anxiety disorders',
        'Schizophrenia disorders (share of population) - Sex: Both - Age: Age-standardized',
        'Depressive disorders (share of population) - Sex: Both - Age: Age-standardized',
        'Anxiety disorders (share of population) - Sex: Both - Age: Age-standardized',
        'Bipolar disorders (share of population) - Sex: Both - Age: Age-standardized',
        'Eating disorders (share of population) - Sex: Both - Age: Age-standardized'
    ]

    print("Please choose the target column for 5-level classification:")
    for i, col in enumerate(potential_target_cols):
        print(f"{i+1}. {col}")

    while True:
        try:
            choice = int(input(f"Enter the number (1-{len(potential_target_cols)}): "))
            if 1 <= choice <= len(potential_target_cols):
                target_col = potential_target_cols[choice - 1]
                print(f"\nYou selected: {target_col}\n")
                break
            else:
                print("Invalid choice. Please enter a number within the range.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    try:
        df = pd.read_csv('Processed Dataset/FinalMerged.csv')
        if 'Contry Name' in df.columns and 'Country Name' not in df.columns:
            df.rename(columns={'Contry Name': 'Country Name'}, inplace=True)

    except FileNotFoundError:
        print("Error: 'Processed Dataset/FinalMerged.csv' not found.")
        return

    if target_col not in df.columns:
        print(f"Error: The selected target column '{target_col}' does not exist in the CSV.")
        return

    df = df.dropna(subset=[target_col])
    if df.empty:
        print(f"Error: No valid data remaining for target '{target_col}' after dropping NaNs.")
        return

    try:
        df['target'] = pd.qcut(df[target_col], q=5, labels=False, duplicates='drop')
    except ValueError as e:
        print(f"Error creating bins with pd.qcut: {e}")
        print(f"Unique non-NaN values in '{target_col}': {df[target_col].nunique()}")
        return

    n_classes = df['target'].nunique()
    if n_classes < 2:
        print(f"Error: Target variable has only {n_classes} unique class after binning. Cannot proceed with classification.")
        return
    print(f"Target variable created with {n_classes} classes (0 to {n_classes-1}).")
    print("Class distribution:\n", df['target'].value_counts(normalize=True).sort_index())

    y = df['target']
    X = df.drop(columns=[target_col, 'target', 'Country Code', 'Country Name'])

    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(include='object').columns.tolist()
    
    # Ensure 'Region' and 'IncomeGroup' are treated as categorical if they are in X
    # and not already in categorical_features (e.g., if they were loaded as numbers)
    potential_cats = ['Region', 'IncomeGroup']
    for pc in potential_cats:
        if pc in X.columns and pc not in categorical_features and pc not in numeric_features:
            # This case might happen if they are object but not caught, or need explicit casting
            X[pc] = X[pc].astype(str) # Ensure they are string for OHE
            categorical_features.append(pc)
        elif pc in X.columns and pc in numeric_features and pc not in categorical_features:
            # If 'Year' or 'IncomeGroup' (if numerically encoded like 1,2,3,4) is numeric but should be categorical
            print(f"Warning: '{pc}' is numeric. If it's categorical, consider one-hot encoding or label encoding.")


    # Example explicit assignment:
    explicit_categorical_features = ['Region', 'IncomeGroup'] 
    numeric_features = [col for col in X.columns if col not in explicit_categorical_features]
    categorical_features = [col for col in explicit_categorical_features if col in X.columns]


    if not numeric_features and not categorical_features:
        print("Error: No features left after selection.")
        return
    
    print(f"Numeric features: {numeric_features}")
    print(f"Categorical features: {categorical_features}")


    # --- Preprocessing Pipelines ---
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Create preprocessor
    transformers_list = []
    if numeric_features:
        transformers_list.append(('num', numeric_transformer, numeric_features))
    if categorical_features:
        transformers_list.append(('cat', categorical_transformer, categorical_features))

    if not transformers_list:
        print("Error: No transformers for preprocessing. Check feature lists.")
        return

    preprocessor = ColumnTransformer(transformers=transformers_list, remainder='drop')


    # Train/test split
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    except ValueError as e:
         print(f"Error during train_test_split: {e}. Check class distribution.")
         return

    # Define models and their hyperparameter grids for GridSearchCV
    models_and_params = {
        'LogisticRegression': (
            LogisticRegression(solver='lbfgs', max_iter=2000, multi_class='auto', random_state=42),
            {'classifier__C': [0.01, 0.1, 1, 10], 'classifier__class_weight': [None, 'balanced']}
        ),
        'RandomForest': (
            RandomForestClassifier(random_state=42),
            {'classifier__n_estimators': [100, 200],
             'classifier__max_depth': [None, 10, 20],
             'classifier__min_samples_split': [2, 5],
             'classifier__class_weight': [None, 'balanced', 'balanced_subsample']}
        ),
        'GradientBoosting': (
            GradientBoostingClassifier(random_state=42),
            {'classifier__n_estimators': [100, 200],
             'classifier__learning_rate': [0.01, 0.1],
             'classifier__max_depth': [3, 5]}
        ),
        'SVC': (
            SVC(probability=True, random_state=42), 
            {'classifier__C': [0.1, 1, 10],
             'classifier__gamma': ['scale', 'auto', 0.01, 0.1],
             'classifier__kernel': ['rbf', 'linear'],
             'classifier__class_weight': [None, 'balanced']}
        ),
        'XGBoost': (
            XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss'),
            {'classifier__n_estimators': [100, 200],
             'classifier__learning_rate': [0.01, 0.1],
             'classifier__max_depth': [3, 5, 7]}
        )
    }

    results = []
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) # For GridSearchCV

    for name, (model, params) in models_and_params.items():
        print(f"\n--- Tuning and Evaluating {name} ---")

        # Pipeline with preprocessor and classifier
        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('classifier', model)])

        # GridSearchCV for hyperparameter tuning
        grid_search = GridSearchCV(pipeline, params, cv=cv_strategy, scoring='f1_weighted', n_jobs=-1, verbose=0)
        grid_search.fit(X_train, y_train)

        print(f"Best parameters for {name}: {grid_search.best_params_}")
        best_model = grid_search.best_estimator_

        y_pred = best_model.predict(X_test)
        
        class_labels = sorted(y.unique())
        cm = confusion_matrix(y_test, y_pred, labels=class_labels)
        cm_df = pd.DataFrame(cm, index=[f'Actual {i}' for i in class_labels],
                             columns=[f'Pred {i}' for i in class_labels])
        print(f"\nConfusion Matrix for {name} (Best Model):\n", cm_df)

        print(f"\nClassification Report for {name} (Best Model):\n",
              classification_report(y_test, y_pred, labels=class_labels, zero_division=0, target_names=[f'Class {i}' for i in class_labels]))

        accuracy = accuracy_score(y_test, y_pred)
        precision_w = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall_w = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1_w = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        results.append({
            'Model': name,
            'Best Params': grid_search.best_params_,
            'Best CV F1 (Weighted)': grid_search.best_score_, 
            'Test Accuracy': accuracy,
            'Test Precision (W)': precision_w,
            'Test Recall (W)': recall_w,
            'Test F1 (W)': f1_w
        })
        print(f"Test Metrics for {name}: Accuracy={accuracy:.4f}, Precision(W)={precision_w:.4f}, Recall(W)={recall_w:.4f}, F1(W)={f1_w:.4f}")


    results_df = pd.DataFrame(results)
    print("\n--- Overall Model Performance (with Hyperparameter Tuning) ---")
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 2000)
    print(results_df.sort_values(by='Test F1 (W)', ascending=False).to_string(index=False))

if __name__ == "__main__":
    main()