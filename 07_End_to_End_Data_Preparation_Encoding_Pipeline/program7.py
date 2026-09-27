import pandas as pd
import numpy as np 


# File 1: User Activity Logs (50 Records)
activity_data = {
    'User_ID': [f'USR_{100+i}' for i in range(50)],
    'Device_ID': [f'DEV_{(i%10)+1}' for i in range(50)],
    'Daily_Listening_Mins': [180, 25, 210, 40, 190, 30, 220, 35, 175, 20, 195, 28, 205, 45, 185, 22, 215, 38, 170, 24, 200, 32, 225, 42, 180, 26, 210, 36, 190, 21, 198, 29, 208, 48, 178, 23, 218, 39, 172, 27, 202, 33, 222, 41, 182, 25, 212, 37, 188, 22],
    'Playlists_Created': [25, 2, 30, 3, 22, 1, 35, 4, 20, 1, 26, 2, 28, 4, 21, 1, 32, 3, 19, 1, 27, 2, 34, 4, 23, 2, 29, 3, 24, 1, 26, 2, 31, 5, 20, 1, 33, 3, 18, 2, 28, 3, 36, 4, 22, 1, 30, 3, 25, 1],
    'Skip_Ratio': [0.12, 0.65, 0.08, 0.58, 0.15, 0.70, 0.05, 0.60, 0.10, 0.75, 0.11, 0.62, 0.09, 0.55, 0.14, 0.68, 0.06, 0.59, 0.13, 0.72, 0.10, 0.64, 0.04, 0.56, 0.12, 0.66, 0.08, 0.57, 0.15, 0.74, 0.09, 0.61, 0.07, 0.54, 0.11, 0.69, 0.05, 0.58, 0.12, 0.63, 0.10, 0.65, 0.04, 0.55, 0.13, 0.67, 0.07, 0.58, 0.14, 0.71],
    'Podcast_Listening_Pct': [0.45, 0.05, 0.50, 0.10, 0.40, 0.02, 0.55, 0.08, 0.38, 0.01, 0.46, 0.04, 0.48, 0.12, 0.41, 0.03, 0.52, 0.09, 0.39, 0.02, 0.47, 0.06, 0.54, 0.11, 0.42, 0.05, 0.49, 0.08, 0.40, 0.01, 0.45, 0.04, 0.51, 0.13, 0.37, 0.02, 0.53, 0.09, 0.38, 0.05, 0.46, 0.07, 0.56, 0.10, 0.43, 0.03, 0.50, 0.08, 0.41, 0.02]
}

# File 2: Subscription Metadata (50 Records)
sub_data = {
    'User_ID': [f'USR_{100+i}' for i in range(50)],
    'Account_Age_Months': [24, 2, 36, 4, 18, 1, 48, 5, 12, 1, 28, 3, 32, 6, 16, 2, 42, 4, 14, 1, 30, 3, 44, 5, 20, 2, 34, 4, 22, 1, 26, 3, 38, 7, 15, 2, 46, 4, 13, 2, 29, 3, 50, 6, 17, 1, 35, 4, 21, 1],
    'Monthly_Fee_USD': [14.99, 0.00, 14.99, 0.00, 14.99, 0.00, 14.99, 0.00, 9.99, 0.00, 14.99, 0.00, 14.99, 0.00, 9.99, 0.00, 14.99, 0.00, 9.99, 0.00, 14.99, 0.00, 14.99, 0.00, 14.99, 0.00, 14.99, 0.00, 14.99, 0.00, 14.99, 0.00, 14.99, 0.00, 9.99, 0.00, 14.99, 0.00, 9.99, 0.00, 14.99, 0.00, 14.99, 0.00, 9.99, 0.00, 14.99, 0.00, 14.99, 0.00]
}

# File 3: Device & Platform Logs (10 Records)
device_data = {
    'Device_ID': [f'DEV_{i}' for i in range(1, 11)],
    'Primary_OS': ['iOS', 'Android', 'iOS', 'Android', 'WebOS', 'iOS', 'Android', 'iOS', 'Android', 'Windows'],
    'App_Version_Code': [4.5, 4.1, 4.5, 4.2, 3.9, 4.5, 4.1, 4.5, 4.2, 4.0]
}

df_activity = pd.DataFrame(activity_data)
df_sub = pd.DataFrame(sub_data)
df_device = pd.DataFrame(device_data)

# TASK 1 — Multi-Table Unsupervised Dataset Merging
# Merge df_activity, df_sub, and df_device using relational keys (User_ID, Device_ID).
df = df_activity.merge(df_sub,how='left',on='User_ID').merge(df_device,how='left',on='Device_ID')
df

# TASK 2 — Paradigm & Label Absence Verification
# Verify shape, column types, and explicitly demonstrate the ABSENCE of a target label (y).
rows,cols  = df.shape
print("Total Combined Records : ",rows)
print("Total Combined Columns : ",cols)
print("Target Vector (y)      : ABSENT (Unsupervised Setup)")
print("Paradigm               : Unsupervised Learning / Clustering")

# TASK 3 — Feature Space Scale & Variance Inspection
# Extract numeric feature set X and inspect feature ranges to justify feature scaling.
X = df[
    [
        'Daily_Listening_Mins',
        'Playlists_Created',
        'Skip_Ratio',
        'Podcast_Listening_Pct',
        'Account_Age_Months'
    ]
]
min_max = df[['Daily_Listening_Mins','Playlists_Created','Skip_Ratio','Podcast_Listening_Pct','Account_Age_Months']].agg(['min','max'])

for i in X.columns:
  mini = X[i].min()
  maxi = X[i].max()
  span = maxi - mini 
  print(
      f"{i:25s} : "
      f"Min = {mini}, "
      f"Max = {maxi}, "
      f"Span = {span}"
  )

print(X.var())
# TASK 4 — Heuristic Baseline Segmentation
# Define heuristic baseline segments:
condition = ((df["Daily_Listening_Mins"]> 100 )& (df['Monthly_Fee_USD']> 0))
X['heuristic'] = np.where(condition,0,1)

cluster_count = X['heuristic'].value_counts()
X['heuristic'].value_counts(normalize=True) * 100

for cluster,count in cluster_count.items():
  percentage = count/len(X) * 100
  if cluster == 0:
    name = 'power listener'
  else: 
    name = 'casual listener'

  print("cluster",cluster,name,count,percentage)

  
# TASK 5 — Cluster Centroid Profiling
# Compute mean metrics for each heuristic cluster across all continuous features.
X.groupby('heuristic').mean()