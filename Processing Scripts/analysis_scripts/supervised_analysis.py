import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def main():

    df = pd.read_csv('Processed Dataset/FinalMerged.csv') 

    # Select target and features
    target_col = 'Depressive disorders (share of population) - Sex: Both - Age: Age-standardized'
    df['target'] = (df[target_col] > df[target_col].median()).astype(int)  # binarize at median

    # Drop non-numeric and identifier columns
    drop_cols = ['Country Code', 'Contry Name', 'Region', 'IncomeGroup', target_col]
    X = df.drop(columns=drop_cols + ['target'])
    X = X.select_dtypes(include=['int64', 'float64'])  # numeric features only
    X = X.fillna(X.median())  # simple median imputation for missing values

    y = df['target']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Define models
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000, solver='lbfgs'),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'SVC': SVC(kernel='rbf', probability=True)
    }

    # Train, predict, evaluate
    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        try:
            y_prob = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_prob)
        except:
            auc = None
        results.append({
            'Model': name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1 Score': f1_score(y_test, y_pred),
            'ROC AUC': auc
        })

    results_df = pd.DataFrame(results)
    print("\nModel performance:\n")
    print(results_df.to_string(index=False))

if __name__ == "__main__":
    main()