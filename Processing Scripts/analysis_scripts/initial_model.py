import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

file_path = 'Processed Dataset/FinalMerged.csv'

try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
    exit()

df.rename(columns={
    'Contry Name': 'Country Name',
    'Life expectancy at birth (years)': 'le_alt',
    'GDP per capita (USD)': 'gdp_pc_alt',
    'Individuals using the Internet (% of population)': 'internet_usage_alt',
    'Unemployment (% of total labor force) (modeled ILO estimate)': 'unemployment_rate',
    'Depressive disorders (share of population) - Sex: Both - Age: Age-standardized': 'depressive_share',
    'DALYs (rate) - Sex: Both - Age: Age-standardized - Cause: Depressive disorders': 'depressive_dalys'
    }, inplace=True)


features = [
    'Year', 'gnipc', 'hdi', 'le', 'eys', 'mys', 'gii', 'loss',
    'internet_usage_alt', 'gdp_pc_alt', 'unemployment_rate',
    'Population density (people per sq. km of land area)',
    'coef_ineq', 'ineq_inc', 'ineq_le', 'ineq_edu'
    ]

target = 'depressive_share'


valid_features = [f for f in features if f in df.columns]
if target not in df.columns:
     print(f"Error: Target column '{target}' not found in the dataset.")
     exit()

relevant_cols = valid_features + [target]
df_model = df[relevant_cols].copy()

for col in valid_features:
    if pd.api.types.is_numeric_dtype(df_model[col]):
        df_model[col] = pd.to_numeric(df_model[col], errors='coerce')
    else:
         print(f"Warning: Feature column '{col}' is not numeric and will be dropped.")
         valid_features.remove(col)

df_model[target] = pd.to_numeric(df_model[target], errors='coerce')

df_model.replace([np.inf, -np.inf], np.nan, inplace=True)
df_model.dropna(subset=valid_features + [target], inplace=True)

if df_model.empty:
    print("Error: No data remaining after cleaning and removing missing values.")
    exit()

X = df_model[valid_features]
y = df_model[target]

if len(X) < 2:
    print("Error: Not enough data points to split into training and testing sets after cleaning.")
    exit()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1, max_depth=10, min_samples_split=5)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("--- Mental Illness Prediction Model Evaluation ---")
print(f"Target Variable: {target}")
print(f"Features Used: {valid_features}")
print(f"Model Algorithm: RandomForestRegressor")
print(f"\nEvaluation Metrics:")
print(f"  Mean Absolute Error (MAE): {mae:.6f}")
print(f"  Mean Squared Error (MSE): {mse:.6f}")
print(f"  Root Mean Squared Error (RMSE): {rmse:.6f}")
print(f"  R-squared (R²): {r2:.6f}")
print("--- End of Evaluation ---")