# HR-PROJECT
# Data Science Analysis 
import pandas as pd

# Load the file to inspect its sheet names and columns
file_path = "HHS_Unaccompanied_Alien_Children_Program (1) (2).xlsx"
xls = pd.ExcelFile(file_path)
print("Sheet names:", xls.sheet_names)

# Read the first sheet to understand the data structure
df = pd.read_excel(xls, sheet_name=xls.sheet_names[0])
print("\nFirst few rows:")
print(df.head())
print("\nColumns and Data Types:")
print(df.info())
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Sort by Date to make it a proper chronological time series
df = df.sort_values('Date').reset_index(drop=True)

# Rename columns for ease of use in python code
df.columns = [
    'Date', 
    'CBP_Apprehended_Intake', 
    'In_CBP_Custody', 
    'Transferred_Out_CBP_HHS_Flow', 
    'In_HHS_Care', 
    'Discharged_From_HHS'
]

# Check data statistics
print(df.describe())

# Check missing dates or continuity
df.set_index('Date', inplace=True)
full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
print(f"\nDate range from {df.index.min()} to {df.index.max()}")
print(f"Expected number of days: {len(full_range)}, Actual observations: {len(df)}")

# Let's perform a simple linear or time-series feature engineering and see how we can structure the forecast
df_reindexed = df.reindex(full_range)
print(f"Missing days after reindexing to full daily sequence: {df_reindexed.isna().sum().to_dict()}")

# Interpolate missing values to build a continuous daily time-series
df_clean = df_reindexed.interpolate(method='linear')
print("Missing days after interpolation:", df_clean.isna().sum().sum())
# Feature Engineering
df_clean['Net_HHS_Pressure'] = df_clean['Transferred_Out_CBP_HHS_Flow'] - df_clean['Discharged_From_HHS']
df_clean['HHS_Care_7D_Mean'] = df_clean['In_HHS_Care'].rolling(window=7).mean()
df_clean['HHS_Care_14D_Mean'] = df_clean['In_HHS_Care'].rolling(window=14).mean()

# Drop NaNs created by rolling features for model training
df_ml = df_clean.dropna()

# Print out some key insights for the paper and presentation
print("Max Care Load:", df_clean['In_HHS_Care'].max())
print("Min Care Load:", df_clean['In_HHS_Care'].min())
print("Average daily flow into HHS:", df_clean['Transferred_Out_CBP_HHS_Flow'].mean())
print("Average daily discharge from HHS:", df_clean['Discharged_From_HHS'].mean())
print("Correlation matrix for In_HHS_Care:")
print(df_clean.corr()['In_HHS_Care'])
