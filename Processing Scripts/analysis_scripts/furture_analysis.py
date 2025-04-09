import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib

def load_and_preprocess_data(file_path, target_column):
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported file format. Please provide a CSV or Excel file.")

        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in the dataset.")

        y = df[target_column]
        X = df.drop(columns=[target_column])

        numerical_features = X.select_dtypes(include=['number']).columns
        categorical_features = X.select_dtypes(include=['object']).columns

        numerical_imputer = SimpleImputer(strategy='mean')
        X[numerical_features] = numerical_imputer.fit_transform(X[numerical_features])

        categorical_imputer = SimpleImputer(strategy='most_frequent')
        X[categorical_features] = categorical_imputer.fit_transform(X[categorical_features])

        X = pd.get_dummies(X, columns=categorical_features, drop_first=True)

        scaler = StandardScaler()
        X[numerical_features] = scaler.fit_transform(X[numerical_features])

        return X, y, scaler

    except Exception as e:
        print(f"Error during data loading and preprocessing: {e}")
        return None, None, None

def train_model(X_train, y_train, model_type='random_forest'):
    if model_type == 'random_forest':
        model = RandomForestRegressor(random_state=42)
    elif model_type == 'gradient_boosting':
        model = GradientBoostingRegressor(random_state=42)
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Choose 'random_forest' or 'gradient_boosting'.")

    model.fit(X_train, y_train)
    print(f"Trained {model_type} model.")
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("Model Evaluation:")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R-squared: {r2:.4f}")
    return mse, r2

def save_model_and_scaler(model, scaler, model_filename="trained_model.joblib", scaler_filename="scaler.joblib"):
    try:
        joblib.dump(model, model_filename)
        joblib.dump(scaler, scaler_filename)
        print(f"Trained model saved as '{model_filename}'")
        print(f"Scaler saved as '{scaler_filename}'")
    except Exception as e:
        print(f"Error saving model and scaler: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a machine learning model on the combined dataset.")
    parser.add_argument("file_path", type=str, nargs="?", default="../../Processed Dataset/FinalMerged.csv",
                        help="Path to the processed dataset file (CSV or Excel).")
    parser.add_argument("target_column", type=str, default="MentalHealthIndex",
                        help="Name of the target variable column.")
    parser.add_argument("--model_type", type=str, default="random_forest", choices=['random_forest', 'gradient_boosting'],
                        help="Type of model to train (random_forest or gradient_boosting).")
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="Proportion of the dataset to use for the test set.")
    parser.add_argument("--random_seed", type=int, default=42,
                        help="Random seed for splitting the data.")
    parser.add_argument("--output_model", type=str, default="trained_model.joblib",
                        help="Filename for saving the trained model.")
    parser.add_argument("--output_scaler", type=str, default="scaler.joblib",
                        help="Filename for saving the scaler.")
    args = parser.parse_args()

    X, y, scaler = load_and_preprocess_data(args.file_path, args.target_column)

    if X is not None and y is not None:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=args.random_seed)

        model = train_model(X_train, y_train, args.model_type)

        if model:
            evaluate_model(model, X_test, y_test)

            save_model_and_scaler(model, scaler, args.output_model, args.output_scaler)
