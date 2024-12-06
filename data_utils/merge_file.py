
import pandas as pd

# Load the CSV files
df_cn = pd.read_csv("data/biology_brain_cn.csv")
df_my = pd.read_csv("data/biology_brain_my.csv")
df = pd.read_csv("data/biology_brain.csv")

# Merge the DataFrames on the 'Index' column
merged_df = pd.merge(df, df_cn[['Index', 'Chinese']], on='Index', how='left')
merged_df = pd.merge(merged_df, df_my[['Index', 'Malay']], on='Index', how='left')

# Save the merged DataFrame to a new CSV file
merged_df.to_csv("data/merged_biology_brain_v1.csv", index=False)
print(f"Number of data in the final merged CSV: {len(merged_df)}")
