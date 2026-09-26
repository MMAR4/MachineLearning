import pandas as pd
# from io import StringIO 
# data = pd.read_csv(StringIO(csv_data))

csv_data ="""Employee_ID,Distance_KM,Travel_Time_Min,Previous_Late_Count,Weather,Transport,Late
E101,5,20,1,Clear,Bus,No
E102,18,65,5,Rain,Bus,Yes
E103,7,30,0,Clear,Car,No
E104,22,75,7,Rain,Bus,Yes
E105,10,40,2,Cloudy,Bike,No"""

rows = [row.split(',') for row in csv_data.split('\n')]
header = rows[0]

data = pd.DataFrame(rows[1:],columns=header)

# TASK 1 — Create the Dataset
# Create the above dataset using Pandas.
print(data)

# TASK 2 — Display Dataset Information
# Display the number of rows, columns, and column names.

no_rows= data.shape[0]
no_columns = data.shape[1]

print('Number of Rows    : ',no_rows)
print('Number of Columns : ',no_columns)

print(*header)

# TASK 3 — Identify Features and Target
# Identify which columns are input features and which column is the target.
features = [col for col in header if col !='Late' and col !='Employee_ID' ]
target = [col for col in header if col == 'Late']

print('Features:',*features)
print("Target: \n" ,*target)

# TASK 4 — Count Late and Non-Late Employees
# Calculate the number of employees who arrived late and who did not arrive late.

# late = (data['Late']=="Yes").sum()
# not_late = (data['Late']=="No").sum()

a,b = data['Late'].value_counts()
print('Late Employees :',b)
print('Not Late Employees :',a)

# TASK 5 — Calculate Average Travel Time
# --------------------------------------
# Calculate the average travel time separately for employees who were late and employees who were not late.
data[['Distance_KM','Travel_Time_Min','Previous_Late_Count']] = data[['Distance_KM','Travel_Time_Min','Previous_Late_Count']].astype(int)
late_avg = data[data["Late"]=="Yes"]['Travel_Time_Min'].mean()
not_late_avg = data[data["Late"]=="No"]['Travel_Time_Min'].mean()


print('Average Travel Time')
print('Late Employees:')
print(late_avg,"minutes")
print('Not Late Employees:')
print(f"{not_late_avg} minutes")


# TASK 6 — Apply a Rule-Based Prediction
import numpy as np 

data['Prediction'] = np.where(
  (data["Distance_KM"] > 15) & (data["Travel_Time_Min"] > 60),
  "Yes",
  "No")


print(data[['Employee_ID','Late',"Prediction"]])

# TASK 7 — Calculate Rule Accuracy

correct = (data['Late'] == data['Prediction']).sum()
total = data['Late'].count()

print('Correct Predictions : ',correct)
print('Total Predictions   : ',total)

print(f"Rule-Based Accuracy : {round(correct/total*100)}%")

# TASK 8 — Predict for a New Employee
emp1 = {
  "Distance_KM" : 20,
  "Travel_Time_Min": 70,
  "Previous_Late_Count" : 3,
  "Weather" : "Rain",
  "Transport" : "Bus",
  "Late" : None
}

data.loc[len(data)] = emp1
data['Prediction']=np.where((data["Distance_KM"]> 15) & (data['Travel_Time_Min']>60),"Yes","No")
data

# TASK 9 — Test Another New Employee

emp2 = {
  'Distance_KM' : 12,
  'Travel_Time_Min' : 55,
  'Previous_Late_Count' : 6,
  'Weather' : 'Rain',
  'Transport' : 'Bus'
}

data.loc[len(data)] = emp2
data['Prediction'] = np.where((data['Distance_KM']>15) & (data["Travel_Time_Min"] > 60),'Yes',"No")
data 

print(f"Employee Distance : {data.iloc[-1]['Distance_KM']} KM")
print(f"Travel Time       : {data.iloc[-1]['Travel_Time_Min']} Minutes")
print(f"Prediction: {data.iloc[-1]['Prediction']}")

# TASK 10 — Rule-Based System vs Machine Learning