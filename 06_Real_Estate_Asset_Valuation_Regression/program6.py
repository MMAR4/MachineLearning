import pandas as pd
import numpy as np

listings_data = {
    'Property_ID': [f'PROP_{100+i}' for i in range(50)],
    'Neighborhood_ID': [f'NGH_{(i%5)+1}' for i in range(50)],
    'Area_SqFt': [2500, 1200, 4000, 1800, 3200, 950, 2800, 1500, 4500, 2100, 3100, 1300, 4200, 1900, 3300, 1000, 4600, 1600, 3400, 2200, 2900, 1400, 4100, 2000, 3500, 1100, 4700, 1700, 3600, 2300, 3000, 1250, 4300, 1850, 3250, 1050, 4400, 1650, 3350, 2150, 2850, 1350, 3950, 1750, 3450, 1150, 4800, 1550, 3700, 2250],
    'Bedrooms': [4, 2, 5, 3, 4, 1, 4, 2, 5, 3, 4, 2, 5, 3, 4, 1, 5, 2, 4, 3, 4, 2, 5, 3, 4, 1, 5, 2, 4, 3, 4, 2, 5, 3, 4, 1, 5, 2, 4, 3, 4, 2, 5, 3, 4, 1, 5, 2, 4, 3],
    'Building_Age_Yrs': [8, 25, 2, 15, 5, 30, 10, 18, 1, 12, 6, 22, 3, 14, 4, 28, 1, 16, 5, 11, 9, 20, 2, 13, 4, 26, 1, 17, 3, 10, 7, 24, 2, 15, 5, 29, 1, 18, 4, 12, 8, 21, 3, 16, 5, 27, 1, 19, 3, 11],
    'Price_USD_M': [1.25, 0.42, 2.85, 0.68, 1.95, 0.29, 1.40, 0.55, 3.20, 0.92, 1.80, 0.48, 2.95, 0.72, 1.90, 0.32, 3.35, 0.58, 2.05, 0.98, 1.48, 0.51, 2.88, 0.78, 2.15, 0.35, 3.45, 0.62, 2.25, 1.05, 1.65, 0.45, 3.05, 0.70, 1.98, 0.31, 3.15, 0.60, 2.10, 0.95, 1.42, 0.49, 2.75, 0.65, 2.20, 0.38, 3.50, 0.56, 2.30, 1.02]
}


# File 2: Neighborhood Analytics (5 Records)
neighborhood_data = {
    'Neighborhood_ID': [f'NGH_{i}' for i in range(1, 6)],
    'Distance_City_KM': [2.5, 18.0, 5.0, 12.0, 1.2],
    'School_Rating': [9.2, 5.5, 8.8, 6.8, 9.6],
    'Crime_Index': [12.0, 45.0, 18.0, 32.0, 8.5]
}

# File 3: Macroeconomic Indicators (50 Records)
macro_data = {
    'Property_ID': [f'PROP_{100+i}' for i in range(50)],
    'Interest_Rate_Pct': [6.5, 6.5, 6.5, 6.5, 6.5, 6.5, 6.5, 6.5, 6.5, 6.5, 6.8, 6.8, 6.8, 6.8, 6.8, 6.8, 6.8, 6.8, 6.8, 6.8, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 6.2, 6.2, 6.2, 6.2, 6.2, 6.2, 6.2, 6.2, 6.2, 6.2, 6.4, 6.4, 6.4, 6.4, 6.4, 6.4, 6.4, 6.4, 6.4, 6.4],
    'Property_Tax_Rate': [0.012, 0.012, 0.012, 0.012, 0.012, 0.012, 0.012, 0.012, 0.012, 0.012, 0.014, 0.014, 0.014, 0.014, 0.014, 0.014, 0.014, 0.014, 0.014, 0.014, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.011, 0.011, 0.011, 0.011, 0.011, 0.011, 0.011, 0.011, 0.011, 0.011, 0.013, 0.013, 0.013, 0.013, 0.013, 0.013, 0.013, 0.013, 0.013, 0.013]
}

df_listings = pd.DataFrame(listings_data)
df_neighborhood = pd.DataFrame(neighborhood_data)
df_macro = pd.DataFrame(macro_data)

# TASK 1 — Multi-Source Relational Data Merging
# Merge df_listings, df_neighborhood, and df_macro using primary/foreign key joins (Property_ID, Neighborhood_ID).
df = df_listings.merge(df_neighborhood,how='left',on='Neighborhood_ID').merge(df_macro,how='left',on='Property_ID')
df


# TASK 2 — Supervised Paradigm Verification
# Verify dataset shape, column names, and confirm target type (Continuous Numerical vs. Discrete).
total_records, cols = df.shape

target_variable = "Price_USD_M"
target_dtype = df[target_variable].dtype

print(f"{'Total Records':<22}: {total_records}")
print(f"{'Total Columns':<22}: {cols}")
print(f"{'Target Variable':<22}: {target_variable}")
print(f"{'Target Data Type':<22}: {target_dtype}")
print(f"{'Supervised Paradigm':<22}: Regression")
print(f"{'Target Type':<22}: Continuous Numerical")


# TASK 3 — Feature Space & Target Structuring
# Extract Feature Matrix (X) and Target Vector (y). Exclude non-predictive identifiers (Property_ID, Neighborhood_ID).
X =  df.drop(columns = ['Property_ID', 'Neighborhood_ID','Price_USD_M'])

for i in X.columns:
  print(i)

y = 'Price_USD_M'


# TASK 4 — Pearson Correlation Matrix Analysis
# Compute Pearson correlation coefficients of all numeric features relative to Price_USD_M.
pearson_correlation = df.corr(numeric_only = True)['Price_USD_M']

mini = pearson_correlation.min()
maxi = pearson_correlation.max()
for features in pearson_correlation.index:
  print(f"{features:<20} :{pearson_correlation[features]:+.3f}")

print(f"Strongest Positive Feature : {pearson_correlation.idxmax()} ({pearson_correlation.max():+.3f})")
print(f'Strongest Negative Feature : {pearson_correlation.idxmin()} {pearson_correlation.min():+.3f}')
df

# TASK 5 — Sub-Group Property Valuation Metrics
# Group dataset into High Value (> $1.5M) and Moderate Value (<= $1.5M) properties. Compute mean Area_SqFt, Distance_City_KM, and School_Rating.
high_value = df[df['Price_USD_M'] > 1.5]
len(high_value)
high_value['Area_SqFt'].mean()

# high_value.agg({'Distance_City_KM':'mean'})
# high_value.agg(Distance_City_KM = ('Distance_City_KM','mean'))
high_value['Distance_City_KM'].mean()
high_value['School_Rating'].mean()


moderate_value =df[df['Price_USD_M']<1.5]
len(moderate_value)
moderate_value['Area_SqFt'].mean()
moderate_value['Distance_City_KM'].mean()
moderate_value['School_Rating'].mean()