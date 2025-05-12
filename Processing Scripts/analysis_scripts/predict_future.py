import pandas as pd
import argparse
import joblib
import logging
from sklearn.impute import SimpleImputer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def load_model_and_scaler(model_path, scaler_path):
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        logging.info("Model and scaler loaded successfully.")
        return model, scaler
    except Exception as e:
        logging.error(f"Error loading model or scaler: {e}")
        return None, None

def read_input_file(file_path):
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
            logging.info("Reading CSV input file...")
        elif file_path.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file_path)
            logging.info("Reading Excel input file...")
        else:
            raise ValueError("Unsupported file format. Use .csv or .xlsx.")
        return df
    except Exception as e:
        logging.error(f"Error reading input file: {e}")
        return None

def impute_data(df):
    try:
        numerical_features = df.select_dtypes(include=['number']).columns
        categorical_features = df.select_dtypes(include=['object']).columns

        numerical_imputer = SimpleImputer(strategy='mean')
        df[numerical_features] = numerical_imputer.fit_transform(df[numerical_features])

        categorical_imputer = SimpleImputer(strategy='most_frequent')
        df[categorical_features] = categorical_imputer.fit_transform(df[categorical_features])

        return df, numerical_features, categorical_features
    except Exception as e:
        logging.error(f"Error during imputation: {e}")
        return None, None, None

def encode_and_scale(df, numerical_features, categorical_features, scaler, expected_features=None):
    try:
        df = pd.get_dummies(df, columns=categorical_features, drop_first=True)

        df[numerical_features] = scaler.transform(df[numerical_features])

        if expected_features is not None:
            for feature in expected_features:
                if feature not in df.columns:
                    df[feature] = 0  # Add missing columns with default 0
            df = df[expected_features]  # Reorder to match training time
        return df
    except Exception as e:
        logging.error(f"Error during encoding or scaling: {e}")
        return None

def predict(model, data):
    try:
        predictions = model.predict(data)
        logging.info("Predictions completed.")
        return predictions
    except Exception as e:
        logging.error(f"Error making predictions: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make predictions using a trained model.")
    parser.add_argument("input_file", type=str, help="Path to the input file (CSV or Excel) containing new data.")
    parser.add_argument("--model_path", type=str, default="trained_model.joblib", help="Path to the trained model file.")
    parser.add_argument("--scaler_path", type=str, default="scaler.joblib", help="Path to the scaler file.")
    parser.add_argument("--output_file", type=str, default="predictions.csv", help="File to save the predictions.")
    parser.add_argument("--expected_features_path", type=str, help="Optional: Path to expected features list (from training).")

    args = parser.parse_args()

    model, scaler = load_model_and_scaler(args.model_path, args.scaler_path)
    if not model or not scaler:
        exit(1)

    df = read_input_file(args.input_file)
    if df is None:
        exit(1)

    df, numerical_features, categorical_features = impute_data(df)
    if df is None:
        exit(1)

    expected_features = None
    if args.expected_features_path:
        try:
            with open(args.expected_features_path, 'r') as f:
                expected_features = f.read().splitlines()
        except Exception as e:
            logging.warning(f"Could not load expected features: {e}")

    processed_data = encode_and_scale(df, numerical_features, categorical_features, scaler, expected_features)
    if processed_data is None:
        exit(1)

    predictions = predict(model, processed_data)
    if predictions is not None:
        output_df = pd.DataFrame(predictions, columns=["Predicted"])
        output_df.to_csv(args.output_file, index=False)
        logging.info(f"Predictions saved to {args.output_file}")
        print(output_df.head())  # Show a preview
