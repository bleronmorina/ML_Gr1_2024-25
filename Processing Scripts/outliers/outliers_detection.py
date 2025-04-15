import pandas as pd
from scipy.stats import zscore
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

file_path = '../../Processed Dataset/FinalMerged.csv'
data = pd.read_csv(file_path)

# Selecting relevant numerical columns (excluding categorical ones)
numerical_columns = [
    "Year", "co2_prod", "coef_ineq", "diff_hdi_phdi", "eys", "eys_f", "eys_m",
    "gdi", "gii", "gni_pc_f", "gni_pc_m", "gnipc", "hdi", "hdi_f", "hdi_m",
    "ihdi", "ineq_edu", "ineq_inc", "ineq_le", "le", "le_f", "le_m", "lfpr_f",
    "lfpr_m", "loss", "mf", "mmr", "mys", "mys_f", "mys_m", "phdi", "pr_f",
    "pr_m", "se_f", "se_m", "Birth rate, crude (per 1,000 people)",
    "Death rate, crude (per 1,000 people)", "Electric power consumption (kWh per capita)",
    "GDP (USD)", "GDP per capita (USD)", "Individuals using the Internet (% of population)",
    "Infant mortality rate (per 1,000 live births)", "Life expectancy at birth (years)",
    "Population density (people per sq. km of land area)",
    "Unemployment (% of total labor force) (modeled ILO estimate)",
    "DALYs (rate) - Sex: Both - Age: Age-standardized - Cause: Depressive disorders",
    "DALYs (rate) - Sex: Both - Age: Age-standardized - Cause: Schizophrenia",
    "DALYs (rate) - Sex: Both - Age: Age-standardized - Cause: Bipolar disorder",
    "DALYs (rate) - Sex: Both - Age: Age-standardized - Cause: Eating disorders",
    "DALYs (rate) - Sex: Both - Age: Age-standardized - Cause: Anxiety disorders"
]

# Step 1: Detect Outliers Using Z-Score
def detect_outliers_zscore(data, columns, threshold=3):
    z_outliers = {}
    for col in columns:
        if col in data.columns:
            z_scores = zscore(data[col].dropna())
            z_outliers[col] = data[abs(z_scores) > threshold]
    return z_outliers

# Step 2: Detect Outliers Using DBSCAN
def detect_outliers_dbscan(data, columns, eps=0.7, min_samples=3):
    dbscan_outliers = {}
    scaler = StandardScaler()
    for col in columns:
        if col in data.columns:
            col_data = data[col].dropna().values.reshape(-1, 1)
            scaled_data = scaler.fit_transform(col_data)
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(scaled_data)
            dbscan_outliers[col] = data.iloc[data[col].dropna().index[labels == -1]]
    return dbscan_outliers

# Step 3: Handle Outliers with Threshold 3
def handle_outliers_z3(data, columns, dbscan_eps=0.7, dbscan_min_samples=3):
    z_outliers = detect_outliers_zscore(data, columns, 3)  # Threshold 3
    dbscan_outliers = detect_outliers_dbscan(data, columns, dbscan_eps, dbscan_min_samples)

    # Combine outliers from both methods
    z_indices = set(pd.concat(z_outliers.values()).index)
    dbscan_indices = set(pd.concat(dbscan_outliers.values()).index)
    all_outlier_indices = z_indices.union(dbscan_indices)

    data['is_outlier'] = data.index.isin(all_outlier_indices)
    flagged_rows = data[data['is_outlier']].copy()
    flagged_rows.to_csv('flagged_outliers.csv', index=False)

    # Remove outliers
    cleaned_data = data[~data.index.isin(z_indices)].copy()
    cleaned_data.drop(columns=['is_outlier'], inplace=True)

    return cleaned_data

# Step 4: Handle Outliers with Threshold 2
def handle_outliers_z2(data, columns, dbscan_eps=0.7, dbscan_min_samples=3):
    z_outliers = detect_outliers_zscore(data, columns, 2)  # Threshold 2
    dbscan_outliers = detect_outliers_dbscan(data, columns, dbscan_eps, dbscan_min_samples)

    # Combine outliers from both methods
    z_indices = set(pd.concat(z_outliers.values()).index)
    dbscan_indices = set(pd.concat(dbscan_outliers.values()).index)
    all_outlier_indices = z_indices.union(dbscan_indices)

    data['is_outlier'] = data.index.isin(all_outlier_indices)
    flagged_rows = data[data['is_outlier']].copy()
    flagged_rows.to_csv('flagged_outliers_z2.csv', index=False)

    # Remove outliers
    cleaned_data = data[~data.index.isin(z_indices)].copy()
    cleaned_data.drop(columns=['is_outlier'], inplace=True)

    return cleaned_data

# Step 5: Handle Outliers with Threshold 1
def handle_outliers_z1(data, columns, dbscan_eps=0.7, dbscan_min_samples=3):
    z_outliers = detect_outliers_zscore(data, columns, 1)  # Threshold 1
    dbscan_outliers = detect_outliers_dbscan(data, columns, dbscan_eps, dbscan_min_samples)

    # Combine outliers from both methods
    z_indices = set(pd.concat(z_outliers.values()).index)
    dbscan_indices = set(pd.concat(dbscan_outliers.values()).index)
    all_outlier_indices = z_indices.union(dbscan_indices)

    data['is_outlier'] = data.index.isin(all_outlier_indices)
    flagged_rows = data[data['is_outlier']].copy()
    flagged_rows.to_csv('flagged_outliers_z1.csv', index=False)

    # Remove outliers
    cleaned_data = data[~data.index.isin(z_indices)].copy()
    cleaned_data.drop(columns=['is_outlier'], inplace=True)

    return cleaned_data

# Step 6: Compare Distributions
def compare_distributions(original_data, cleaned_data, columns):
    for col in columns:
        if col in original_data.columns and col in cleaned_data.columns:
            plt.figure(figsize=(10, 5))
            sns.kdeplot(original_data[col], label="Original", color='red', fill=True, alpha=0.4)
            sns.kdeplot(cleaned_data[col], label="Cleaned", color='blue', fill=True, alpha=0.4)
            plt.title(f"Comparison of {col} Distributions (Original vs Cleaned)")
            plt.xlabel(col)
            plt.ylabel("Density")
            plt.legend()
            plt.show()

if __name__ == "__main__":
    # Test with Z-Score Threshold 3
    cleaned_data_z3 = handle_outliers_z3(data.copy(), numerical_columns)
    compare_distributions(data, cleaned_data_z3, numerical_columns)
    cleaned_data_z3.to_csv('../../Processed Dataset/dataset_cleaned_03.csv', index=False)

    # Test with Z-Score Threshold 2
    # cleaned_data_z2 = handle_outliers_z2(data.copy(), numerical_columns)
    # compare_distributions(data, cleaned_data_z2, numerical_columns)
    # cleaned_data_z2.to_csv('dataset_cleaned_z2.csv', index=False)

    # Test with Z-Score Threshold 1
    # cleaned_data_z1 = handle_outliers_z1(data.copy(), numerical_columns)
    # compare_distributions(data, cleaned_data_z1, numerical_columns)
    # cleaned_data_z1.to_csv('dataset_cleaned_z1.csv', index=False)
