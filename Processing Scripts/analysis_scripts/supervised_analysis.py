import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

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
    except FileNotFoundError:
        print("Error: 'Processed Dataset/FinalMerged.csv' not found.")
        print("Please make sure the dataset file is in the correct directory.")
        return 

    if target_col not in df.columns:
        print(f"Error: The selected target column '{target_col}' does not exist in the CSV.")
        return

    df = df.dropna(subset=[target_col])
    if df.empty:
        print(f"Error: No valid data remaining for the target column '{target_col}' after dropping NaNs.")
        return

    # --- Target Variable Creation (5 Levels) ---
    # Used qcut to create 5 bins (quintiles) based on the target column's distribution
    # labels=False assigns integers 0 through 4 to the bins
    try:
        df['target'] = pd.qcut(df[target_col], q=5, labels=False, duplicates='drop')
    except ValueError as e:
        print(f"Error creating bins with pd.qcut: {e}")
        print("This might happen if the target column has too few unique values to create 5 distinct quantiles.")
        print(f"Unique non-NaN values in '{target_col}': {df[target_col].nunique()}")
        return

    n_classes = df['target'].nunique()
    print(f"Target variable created with {n_classes} classes (0 to {n_classes-1}) based on quantiles.")
    print("Class distribution:\n", df['target'].value_counts(normalize=True).sort_index())


    # --- Feature Preparation ---
    # Drop non-numeric, identifier columns, and the *original* target column
    drop_cols = ['Country Code', 'Contry Name', 'Region', 'IncomeGroup', target_col]
    X = df.drop(columns=drop_cols + ['target'])
    X = X.select_dtypes(include=['int64', 'float64'])  # numeric features only

    if X.shape[1] == 0:
        print("Error: No numeric features left after dropping specified columns.")
        return

    X = X.fillna(X.median()) 

    y = df['target']

    # Check if there's enough data for splitting
    if len(X) < 10: # Arbitrary small number check
        print("Error: Not enough data points to perform train/test split.")
        return

    # Train/test split
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    except ValueError as e:
         print(f"Error during train_test_split: {e}")
         print("This often happens if a class has too few members (e.g., only 1) for stratification.")
         print("Check the class distribution printed above.")
         return

    # Feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Define models (These models inherently support multi-class)
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000, solver='lbfgs'), 
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'SVC': SVC(kernel='rbf', probability=False)
    }

    # Train, predict, evaluate
    results = []
    class_labels = sorted(y.unique()) # Here we get the actual class labels (0, 1, 2, 3, 4)

    for name, model in models.items():
        print(f"\n--- Training and Evaluating {name} ---")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # --- Confusion Matrix (Multi-class) ---
        cm = confusion_matrix(y_test, y_pred, labels=class_labels)
        cm_df = pd.DataFrame(cm,
                     index=[f'Actual {i}' for i in class_labels],
                     columns=[f'Pred {i}' for i in class_labels])
        print(f"\nConfusion Matrix for {name}:\n", cm_df, "\n")

        # --- Multi-class Metrics ---
        # Used 'weighted' average to account for potential class imbalance
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        results.append({
            'Model': name,
            'Accuracy': accuracy,
            'Precision (Weighted)': precision,
            'Recall (Weighted)': recall,
            'F1 Score (Weighted)': f1
        })
        print(f"Metrics for {name}: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")


    results_df = pd.DataFrame(results)
    print("\n--- Overall Model Performance ---")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(results_df.to_string(index=False))

if __name__ == "__main__":
    main()