import pandas as pd 
import numpy as np 

tx_data = {
    'Tx_ID': [f'TX_{1000+i}' for i in range(50)],
    'Account_ID': [f'ACC_{(i%10)+101}' for i in range(50)],
    'Terminal_ID': [f'TRM_{(i%8)+501}' for i in range(50)],
    'Tx_Amount': [15.2, 1250.0, 45.0, 3200.0, 8.5, 980.0, 2100.0, 12.0, 4500.0, 65.0, 1100.0, 25.0, 2800.0, 80.0, 1750.0, 5.0, 3900.0, 18.0, 1300.0, 95.0, 2400.0, 30.0, 4100.0, 40.0, 1600.0, 15.0, 3100.0, 55.0, 2200.0, 10.0, 1450.0, 70.0, 4800.0, 22.0, 1900.0, 12.0, 2600.0, 85.0, 3300.0, 35.0, 1200.0, 60.0, 4200.0, 18.0, 1800.0, 45.0, 2900.0, 25.0, 2100.0, 90.0],
    'Tx_Hour': [14, 2, 11, 3, 16, 1, 4, 18, 23, 10, 2, 15, 1, 12, 3, 19, 4, 13, 2, 9, 3, 17, 1, 11, 4, 20, 2, 8, 3, 15, 1, 14, 4, 18, 2, 12, 3, 10, 1, 16, 2, 11, 4, 21, 3, 9, 2, 13, 1, 7],
    'Is_Fraud': [0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
}

acc_data = {
    'Account_ID': [f'ACC_{i}' for i in range(101, 111)],
    'Avg_Monthly_Tx_Vol': [1200.0, 300.0, 1500.0, 400.0, 2000.0, 250.0, 500.0, 1800.0, 350.0, 1100.0],
    'Account_Age_Months': [48, 6, 60, 3, 84, 2, 12, 72, 4, 36]
}

trm_data = {
    'Terminal_ID': [f'TRM_{i}' for i in range(501, 509)],
    'Terminal_Risk_Score': [0.1, 0.8, 0.2, 0.9, 0.15, 0.75, 0.85, 0.05],
    'Is_Foreign_Location': [0, 1, 0, 1, 0, 1, 1, 0]
}

blk_data = {
    'Tx_ID': [f'TX_{1000+i}' for i in range(50)],
    'High_Risk_IP_Flag': [0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
}

df_tx = pd.DataFrame(tx_data)
df_acc = pd.DataFrame(acc_data)
df_trm = pd.DataFrame(trm_data)
df_blk = pd.DataFrame(blk_data)


# TASK 1 — Enterprise 4-Table Relational Merge: Perform sequential inner/left joins on df_tx, df_acc, df_trm, and df_blk to build a single master fraud analytical table. 
df = df_tx.merge(df_acc,how='left',on="Account_ID").merge(df_trm,how='left',on='Terminal_ID').merge(df_blk,how='left',on='Tx_ID')

df 


# TASK 2 — Schema Validation: Output dataset dimensions, inspect column data types, and check missing value distributions. 
df.info()
df.isna().sum()
# numerical = df.select_dtypes(include='number').columns
# categorical = df.select_dtypes(exclude='number').columns
# df.notna()

# TASK 3 — Feature Space Definition: Classify inputs into predictive features versus operational identifiers (Tx_ID, Account_ID, Terminal_ID). Explicitly specify binary target Is_Fraud.  
X = df.drop(columns = ['Tx_ID','Account_ID', 'Terminal_ID','Is_Fraud'])
y = df['Is_Fraud']

# TASK 4 — Fraud Class Aggregation: Compute class balances (0 vs. 1) and calculate median Tx_Amount and Terminal_Risk_Score per class.  
is_fraud_median = df.groupby('Is_Fraud')[['Tx_Amount','Terminal_Risk_Score']].median()
tx_amount_0, terminal_0 = is_fraud_median.loc[0]
tx_amount_1, terminal_1 = is_fraud_median.loc[1]

df["Is_Fraud"].value_counts()
class_distribution = df["Is_Fraud"].value_counts(normalize=True) * 100


# TASK 5 — Rule-Based Fraud System Simulation: Apply security rule: IF (Tx_Amount > 1000 AND Tx_Hour IN [1,2,3,4]) OR High_Risk_IP_Flag == 1 THEN Prediction = 1 ELSE 0.  
df["Prediction"] = np.where((df["Tx_Amount"]> 1000 ) & (df['Tx_Hour'].isin([1,2,3,4])) | (df['High_Risk_IP_Flag'] == 1),1,0)
df

# TASK 6 — Quantitative Fraud Detection Performance: Compute accuracy, correct detections, false alarms, and missed fraud occurrences.  
correct = (df['Prediction'] == df['Is_Fraud']).sum()
total = df["Is_Fraud"].count()
false_pos = ((df['Prediction']==1) & (df['Is_Fraud'] ==0)).sum()
false_neg = ((df['Prediction']==0) & (df['Is_Fraud'] ==1)).sum()
accuracy = correct/total * 100
# TASK 7 — Train vs Predict Lifecycle Blueprint: Format 5 records representing live transaction stream (excluding target Is_Fraud). Explain how training phase differs from live prediction phase. 
live_transactions = df.drop(columns=["Prediction","Is_Fraud"]).head(5)