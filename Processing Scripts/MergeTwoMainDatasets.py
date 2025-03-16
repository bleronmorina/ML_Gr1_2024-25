import pandas as pd

# Load the CSV files
df1 = pd.read_csv("Unprocessed Datasets/World Economics Indicator Dataset/World_Economics_Indicator_Cleaned.csv")
df2 = pd.read_csv("Processed Dataset/1&2_merged_data.csv")

# Perform INNER JOIN on CountryCode and Year
merged_df = pd.merge(df1, df2, left_on=["Country Code", "Year"], right_on=["Code", "Year"], how="inner")

merged_df.drop(columns=["Code", "Entity"], inplace=True)

# Save the merged result
merged_df.to_csv("Processed Dataset/FinalMerged.csv", index=False)

# Display the first few rows
print(merged_df.head())
