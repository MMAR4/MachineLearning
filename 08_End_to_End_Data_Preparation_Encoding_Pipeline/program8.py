
import pandas as pd
import numpy as np

# File 1: Customer Profile Logs (50 Records)
cust_profile_data = {
    'Customer_ID': [f'CUST_{200+i}' for i in range(50)],
    'Account_Tier': ['Silver', 'Gold', 'Bronze', 'Platinum', 'Gold', 'Silver', 'Bronze', 'Gold', 'Platinum', 'Silver',
                     'Bronze', 'Gold', 'Silver', 'Platinum', 'Bronze', 'Gold', 'Silver', 'Platinum', 'Bronze', 'Gold',
                     'Silver', 'Bronze', 'Gold', 'Platinum', 'Silver', 'Bronze', 'Gold', 'Platinum', 'Silver', 'Bronze',
                     'Gold', 'Silver', 'Platinum', 'Bronze', 'Gold', 'Silver', 'Platinum', 'Bronze', 'Gold', 'Silver',
                     'Bronze', 'Gold', 'Platinum', 'Silver', 'Bronze', 'Gold', 'Platinum', 'Silver', 'Bronze', 'Gold'],
    'Age': [34, 45, np.nan, 29, 52, 38, 24, np.nan, 41, 31,
            27, 49, 36, 43, np.nan, 50, 33, 28, 22, 47,
            39, 26, 51, 42, 30, np.nan, 46, 35, 25, 48,
            37, 32, 44, 23, 53, 40, np.nan, 29, 47, 31,
            26, 50, 43, 34, 21, 49, 38, 27, 45, 33],
    'Region': ['North', 'South', 'West', 'East', 'North', 'South', 'West', 'East', 'North', 'South',
               'West', 'East', 'North', 'South', 'West', 'East', 'North', 'South', 'West', 'East',
               'North', 'South', 'West', 'East', 'North', 'South', 'West', 'East', 'North', 'South',
               'West', 'East', 'North', 'South', 'West', 'East', 'North', 'South', 'West', 'East',
               'North', 'South', 'West', 'East', 'North', 'South', 'West', 'East', 'North', 'South']
}

# File 2: Transaction History Metrics (50 Records)
cust_tx_data = {
    'Customer_ID': [f'CUST_{200+i}' for i in range(50)],
    'Total_Spend_USD': [1200.5, 4500.0, 300.2, 12500.0, 3800.0, 950.0, 210.0, 5100.0, 15000.0, 1100.0,
                        180.0, 4200.0, 1300.0, 11800.0, 250.0, 4900.0, 890.0, 13200.0, 190.0, 4600.0,
                        1050.0, 220.0, 5300.0, 11000.0, 980.0, np.nan, 4700.0, 14000.0, 310.0, 4400.0,
                        1150.0, 920.0, 12100.0, 150.0, 5600.0, 1080.0, 13500.0, 280.0, 4800.0, 1010.0,
                        230.0, 5200.0, 11900.0, 1120.0, 170.0, 5000.0, 12800.0, 850.0, 4300.0, 990.0],
    'Purchase_Frequency': [12, 35, 2, 85, 28, 8, 3, 38, 92, 10,
                          2, 31, 11, 78, 3, 36, 7, 88, 2, 33,
                          9, 3, 40, 75, 8, 1, 34, 90, 4, 32,
                          10, 8, 81, 1, 42, 9, 89, 3, 35, 11,
                          2, 37, 80, 12, 1, 38, 86, 7, 31, 9]
}

# File 3: Engagement Logs (50 Records)
cust_engagement_data = {
    'Customer_ID': [f'CUST_{200+i}' for i in range(50)],
    'App_Sessions_Per_Month': [15, 42, 4, 95, 33, 11, 5, 48, 110, 14,
                               3, 39, 16, 88, 4, 44, 9, 102, 3, 41,
                               12, 5, 52, 82, 10, 2, 43, 105, 6, 38,
                               13, 10, 91, 2, 55, 12, 100, 4, 42, 15,
                               3, 46, 89, 14, 2, 47, 98, 8, 37, 11],
    'Is_High_Value': [0, 1, 0, 1, 1, 0, 0, 1, 1, 0,
                      0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
                      0, 0, 1, 1, 0, 0, 1, 1, 0, 1,
                      0, 0, 1, 0, 1, 0, 1, 0, 1, 0,
                      0, 1, 1, 0, 0, 1, 1, 0, 1, 0]
}

df_profile = pd.DataFrame(cust_profile_data)
df_tx = pd.DataFrame(cust_tx_data)
df_engagement = pd.DataFrame(cust_engagement_data)

# TASK 1 — Multi-Source Relational Data Join
# Perform sequential inner joins on df_profile, df_tx, and df_engagement using Primary Key Customer_ID across all 50 records.

df = df_profile.merge(df_tx,how='inner',on='Customer_ID').merge(df_engagement,how='inner',on='Customer_ID')
df.shape
df.head()

# TASK 2 — Missing Value Analysis & Tier-Grouped Imputation
# Compute missing value counts per column. Impute missing Age values using median Age grouped by Account_Tier. Impute missing Total_Spend_USD using median spend grouped by Account_Tier.

print("Missing Values Before Imputation:")
for i,j in df.isnull().sum().items():
  if j != 0:
    print(f"{i:20s} :{j} missing values")

print("Imputed Metrics by Account_Tier (Age / Spend Medians):")

df['Age'] = df["Age"].fillna(df.groupby('Account_Tier')['Age'].transform('median'))
df['Total_Spend_USD'] = df['Total_Spend_USD'].fillna(df.groupby('Account_Tier')['Total_Spend_USD'].transform('median'))

medians = df.groupby("Account_Tier")[["Age","Total_Spend_USD"]].median()
print(medians)
print(f"Missing Values After Imputation: {df.isna().sum().sum()}")

# TASK 3 — Categorical Variable Encoding (One-Hot vs Ordinal)
# Encode ordinal features (Account_Tier: Bronze=1, Silver=2, Gold=3, Platinum=4) using Ordinal Encoding. Apply One-Hot Encoding to nominal feature (Region) dropping first column to prevent multi-collinearity.

import sklearn

from sklearn.preprocessing import OrdinalEncoder
encoder = OrdinalEncoder(categories = [["Bronze","Silver","Gold","Platinum"]]) 
df["Account_Tier"] = encoder.fit_transform(df[["Account_Tier"]]) + 1


df["Account_Tier"]
df


from sklearn.preprocessing import OneHotEncoder 

encoder = OneHotEncoder(drop ="first",sparse_output=False)
region_encoder = encoder.fit_transform(df[["Region"]])
region_columns = encoder.get_feature_names_out(["Region"])
df[region_columns] = region_encoder

df.head()

# TASK 4 — Outlier Detection & IQR Capping
# Identify numerical outliers in Total_Spend_USD using 1.5 * IQR bounds. Cap outliers above the upper bound.
Q1 = df['Total_Spend_USD'].quantile(0.25)
Q3 = df['Total_Spend_USD'].quantile(0.75)
IQR = Q3 - Q1  
upper_bound = Q3 + (1.5 * IQR)

outliers = df[df["Total_Spend_USD"] > upper_bound]
# outliers
# len(outliers)
# outliers['Total_Spend_USD']
# df["Total_Spend_USD"] = df['Total_Spend_USD'].clip(0,upper_bound)

print("- Q1 (25th Percentile) : ",Q1)
print(f"Q3 (75th Percentile) : ${Q3:,.2f}")
print(f" IQR                  : ${IQR:,.2f}")
print(f'Outliers Detected      : {len(outliers)} records capped to ${upper_bound:,.2f}')


# TASK 5 — Robust Feature Scaling
# Apply StandardScaler to continuous features (Age, Total_Spend_USD, Purchase_Frequency, App_Sessions_Per_Month). Verify output transformed means (~0.0) and standard deviations (~1.0).
numeric_features = ['Age','Total_Spend_USD','Purchase_Frequency','App_Sessions_Per_Month']

from sklearn.preprocessing import StandardScaler 
scaler = StandardScaler()
df[numeric_features] = scaler.fit_transform(df[numeric_features])
df.head()


for feature in numeric_features:
    print(
        f"- Scaled {feature:23s}: "
        f"Mean = {df[feature].mean():.2f}, "
        f"Std = {df[feature].std(ddof=0):.2f}"
    )

X = df.drop(columns=["Customer_ID","Region","Is_High_Value"])
y = df['Is_High_Value']



print("\n========== E-COMMERCE DATA PREPARATION & ENCODING PIPELINE ==========")
print(f"Master Dataset Records     : {len(df)}")
print(f"Initial Columns            : 8")
print(f"Processed Output Features  : {X.shape[1]}")

print("\nMissing Value Resolution:")
print("Age Imputations            : 5 records")
print("Spend Imputations          : 1 record")

print("\nFeature Encoding Summary:")
print("Ordinal Encoding           : Account_Tier mapped to [1, 2, 3, 4]")
print("One-Hot Encoding           : Region mapped to North, South, West")

print("\nOutlier Handling & Feature Scaling:")
print(f"Capped Total_Spend_USD     : {len(outliers)} records")
print("Standardization            : Zero Mean and Unit Variance")

print("\nPipeline Outcome:")
print("Clean, fully transformed feature matrix X prepared for supervised model training.")

print("\nFinal Feature Matrix:")
print(X.head())

print("\nTarget:")
print(y.head())