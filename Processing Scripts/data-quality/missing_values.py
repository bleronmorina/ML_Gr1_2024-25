import pandas as pd
import argparse

def check_missing_values(file_path):
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file_path)
        else:
            print("Unsupported file format. Please provide a CSV or Excel file.")
            return

        missing_values = df.isnull().sum()
        total_missing = missing_values.sum()

        if total_missing == 0:
            print("No missing values found in the dataset.")
        else:
            print("Missing values found in the dataset:")
            print(missing_values[missing_values > 0])
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check for missing values in a dataset.")
    parser.add_argument("file_path", type=str, nargs="?", default="../../Processed Dataset/FinalMerged.csv",
                        help="Path to the dataset file (CSV or Excel).")
    args = parser.parse_args()
    check_missing_values(args.file_path)
