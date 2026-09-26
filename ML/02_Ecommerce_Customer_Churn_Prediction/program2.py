import pandas as pd
from io import StringIO

raw_data = """Customer_ID,Account_Age_Days,Monthly_Spend,Support_Calls,Inactivity_Days,Churn
C101,120,45.5,4,35,Yes
C102,450,120.0,0,5,No
C103,90,25.0,5,40,Yes
C104,300,85.0,1,12,No
C105,210,60.0,2,20,No
C106,60,15.0,6,45,Yes
C107,500,200.0,1,8,No"""

data = pd.read_csv(StringIO(raw_data))


# TASK 1 — Create the DatasetCreate the above dataset using Pandas. 
print(data)

# TASK 2 — Display Dataset Information
rows, columns = data.shape

print('Number of Rows    : ',rows)
print("Number of Columns : ",columns)

print("Columns:")
for col in data.columns:
  print(col)

# TASK 3 — Identify Features and Target
print('Features:')
features = ['Account_Age_Days','Monthly_Spend','Support_Calls','Inactivity_Days']
target = ['Churn']

for i in features:
  print(i)
print("Target:")
print(*target)

# TASK 4 — Count Churned and Retained Customers
no,yes =data["Churn"].value_counts()
print(f"Churn \nNo {no}\nYes {yes}")

# TASK 5 — Calculate Average Inactivity Days
print("Average Inactivity Days:")

ch_customer = data[data['Churn'] =="Yes"]["Inactivity_Days"].mean()
re_customer = data[data['Churn']=="No"]["Inactivity_Days"].mean()
print(f"Churned Customers   : {ch_customer} days")
print(f"Retained Customers : {re_customer} days")

# TASK 6 — Apply a Rule-Based Prediction
import numpy as np
data["Prediction"] = np.where((data["Inactivity_Days"]>30) & (data ["Monthly_Spend"]<50), "Yes","No")

print(data[["Customer_ID","Churn","Prediction"]])

# TASK 7 — Calculate Rule Accuracy  
correct = (data["Churn"] == data["Prediction"]).sum()
total = data['Churn'].count()
rule = correct/total

print(f"Correct Predictions : {correct}")
print(f"Total Predictions   : {total}")
print(f"Rule-Based Accuracy : {round(rule*100)}%")



# TASK 8 — Test New Customer Scenarios 
new_data= pd.DataFrame({
  "Inactivity_Days":[32,28],
  "Monthly_Spend":[150,30],
  "Support_Calls": [6,8]
}
)
# df = pd.concat([data,new_data],ignore_index=True)
new_data["Prediction"]=np.where((new_data["Inactivity_Days"] >30) & (new_data["Monthly_Spend"] <50),"Yes","No")
new_data
print(f"Customer A Prediction : {new_data.loc[0,'Prediction']}")
print(f"Customer B Prediction : {new_data.loc[1,'Prediction']}")


# TASK 9 — Identify Features & Labels in Real-World Context

len(data)
