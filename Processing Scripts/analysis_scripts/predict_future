import pandas as pd
import argparse
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def load_model_and_scaler(model_path, scaler_path):
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        print("Model and scaler loaded successfully.")
        return model, scaler
    except Exception as e:
        print(f"Error loading model or scaler: {e}")
        return None, None

def preprocess_input_data(file_path, scaler):
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
            print("Reading CSV input file...")
        elif file_path.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file_path)
            print("Reading Excel input file...")
        else:
            raise ValueError("Unsupported file format. Use .csv or .xlsx.")

        numerical_features = df.select_dtypes(include=['number']).columns
        categorical_features = df.select_dtypes(include=['object']).columns

        numerical_imputer = SimpleImputer(strategy='mean')
        df[numerical_features] = numerical_imputer.fit_transform(df[numerical_features])

        categorical_imputer = SimpleImputer(strategy='most_frequent')
        df[categorical_features] = categorical_imputer.fit_transform(df[categorical_features])

        df = pd.get_dummies(df, columns=categorical_features, drop_first=True)

        df[numerical_features] = scaler.transform(df[numerical_features])

        return df
    except Exception as e:
        print(f"Error preprocessing input data: {e}")
        return None

def predict(model, data):
    try:
        predictions = model.predict(data)
        print("Predictions completed.")
        return predictions
    except Exception as e:
        print(f"Error making predictions: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make predictions using a trained model.")
    parser.add_argument("input_file", type=str, help="Path to the input file (CSV or Excel) containing new data.")
    parser.add_argument("--model_path", type=str, default="trained_model.joblib", help="Path to the trained model file.")
    parser.add_argument("--scaler_path", type=str, default="scaler.joblib", help="Path to the scaler file.")
    parser.add_argument("--output_file", type=str, default="predictions.csv", help="File to save the predictions.")

    args = parser.parse_args()

    model, scaler = load_model_and_scaler(args.model_path, args.scaler_path)

    if model and scaler:
        data = preprocess_input_data(args.input_file, scaler)
        if data is not None:
            predictions = predict(model, data)
            if predictions is not None:
                output_df = pd.DataFrame(predictions, columns=["Predicted"])
                output_df.to_csv(args.output_file, index=False)
                print(f"Predictions saved to {args.output_file}")
