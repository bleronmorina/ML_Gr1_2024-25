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
            print("reading csv")
        elif file_path.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file_path)
            print("reading xlsx")
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
        print("Error fetching..")
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

def compare_models(X_train, y_train, X_test, y_test):
    """
    Trains and evaluates both RandomForestRegressor and GradientBoostingRegressor models,
    and returns their evaluation metrics for comparison.
    """
    results = {}

    for model_type in ['random_forest', 'gradient_boosting']:
        model = train_model(X_train, y_train, model_type=model_type)
        mse, r2 = evaluate_model(model, X_test, y_test)
        results[model_type] = {'MSE': mse, 'R2': r2}

    return results

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("Model Evaluation:")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R-squared: {r2:.4f}")
    print(f"F-squared: {r2:.4f}")

    return mse, r2

def save_model_and_scaler(model, scaler, model_filename="trained_model.joblib", scaler_filename="scaler.joblib"):
    try:
        joblib.dump(model, model_filename)
        joblib.dump(scaler, scaler_filename)
        print(f"Trained model saved as '{model_filename}'")
        print(f"Scaler saved as '{scaler_filename}'")
    except Exception as e:
        print(f"Error saving model and scaler: {e}")

def calculate_rmse(y_true, y_pred):
    """
    Calculates the Root Mean Squared Error (RMSE).
    """
    from sklearn.metrics import mean_squared_error
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    return rmse

def test_data(y_true, y_pred):
    """
    Testing data.
    """
    from sklearn.metrics import test_data
    coef = training(y_true, y_pred, squared=False)
    return coef

def compare_models_with_rmse(X_train, y_train, X_test, y_test):
    """
    Trains and evaluates both RandomForestRegressor and GradientBoostingRegressor models,
    and returns their evaluation metrics for comparison, including RMSE.
    """
    results = {}

    for model_type in ['random_forest', 'gradient_boosting']:
        model = train_model(X_train, y_train, model_type=model_type)
        mse, r2 = evaluate_model(model, X_test, y_test)
        
        # Calculate RMSE
        y_pred = model.predict(X_test)
        rmse = calculate_rmse(y_test, y_pred)

        results[model_type] = {'MSE': mse, 'R2': r2, 'RMSE': rmse}

    return results


def plot_feature_importance(model, feature_names, top_n=10):
    """
    Plots the top N most important features from a tree-based model.

    Parameters:
    - model: Trained tree-based model (e.g., RandomForestRegressor)
    - feature_names: List of feature names
    - top_n: Number of top features to plot
    """
    try:
        importances = model.feature_importances_
        indices = importances.argsort()[::-1][:top_n]
        plt.figure(figsize=(10, 6))
        plt.barh(range(top_n), importances[indices][::-1], align="center")
        plt.yticks(range(top_n), [feature_names[i] for i in indices][::-1])
        plt.xlabel("Feature Importance")
        plt.title(f"Top {top_n} Feature Importances")
        plt.tight_layout()
        plt.show()
    except AttributeError:
        print("Model does not support feature importances.")
    except Exception as e:
        print(f"Error plotting feature importances: {e}")

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

        # Call the new model comparison function
        comparison_results = compare_models_with_rmse(X_train, y_train, X_test, y_test)
        print("Model Comparison Results:")
        print(comparison_results)

        # Choose the model for further processing (you can select the best based on the results)
        best_model_type = 'random_forest' if comparison_results['random_forest']['RMSE'] < comparison_results['gradient_boosting']['RMSE'] else 'gradient_boosting'
        model = train_model(X_train, y_train, best_model_type)

        # Evaluate and save the best model
        mse, r2 = evaluate_model(model, X_test, y_test)
        save_model_and_scaler(model, scaler, args.output_model, args.output_scaler)
