import pandas as pd
import os

dalys_path = "Unprocessed Datasets/Mental Health Dataset/2- burden-disease-from-each-mental-illness(1).csv"
share_path = "Unprocessed Datasets/Mental Health Dataset/1- mental-illnesses-prevalence.csv"
output_path = "Processed Dataset/1&2_merged_data.csv"

df_dalys = pd.read_csv(dalys_path, encoding="utf-8")
df_share = pd.read_csv(share_path, encoding="utf-8")

print("DALYs Columns:", df_dalys.columns)
print("Share Columns:", df_share.columns)

df_merged = pd.merge(df_dalys, df_share, on=["Entity", "Code", "Year"], how="inner")

output_dir = os.path.dirname(output_path)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

df_merged.to_csv(output_path, index=False, encoding="utf-8")

print(df_merged.head())
