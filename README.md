# Big_Mart_Sales
Big Mart Sales Prediction Project
This project aims to predict the sales of items in different outlets of Big Mart using various machine learning techniques. The dataset includes information on item attributes and outlet characteristics. The primary goal is to develop a predictive model using XGBoost Regressor to estimate the sales.

Table of Contents
Importing the Dependencies
Data Collection and Processing
Handling Missing Values
Data Analysis
Data Pre-Processing
Label Encoding
Splitting Features and Target
Splitting the Data into Training and Testing Data
Machine Learning Model Training
Evaluation
Importing the Dependencies
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics
```
Data Collection and Processing
Loading the Dataset
```python
big_mart_data = pd.read_csv('/content/Train.csv')
big_mart_data.head()
```
Dataset Overview
Number of data points & features: 8523 data points, 12 features
Dataset Information:
Item_Identifier: Unique product ID
Item_Weight: Weight of the product
Item_Fat_Content: Fat content of the product
Item_Visibility: Percentage of total display area allocated to the product
Item_Type: Category to which the product belongs
Item_MRP: Maximum Retail Price (list price) of the product
Outlet_Identifier: Unique store ID
Outlet_Establishment_Year: The year the outlet was established
Outlet_Size: The size of the outlet (small, medium, high)
Outlet_Location_Type: Type of city in which the outlet is located
Outlet_Type: Type of outlet (grocery store or supermarket)
Item_Outlet_Sales: Sales of the product in the particular store
Checking for Missing Values
```python
big_mart_data.isnull().sum()
```
Handling Missing Values
Filling Missing Values in "Item_Weight"
```python
big_mart_data['Item_Weight'].fillna(big_mart_data['Item_Weight'].mean(), inplace=True)
```
Filling Missing Values in "Outlet_Size"
```python
mode_of_Outlet_size = big_mart_data.pivot_table(values='Outlet_Size', columns='Outlet_Type', aggfunc=(lambda x: x.mode()[0]))
miss_values = big_mart_data['Outlet_Size'].isnull()
big_mart_data.loc[miss_values, 'Outlet_Size'] = big_mart_data.loc[miss_values, 'Outlet_Type'].apply(lambda x: mode_of_Outlet_size[x])
```
Verifying Missing Values
```python
big_mart_data.isnull().sum()
```
Data Analysis
Descriptive Statistics
```python
big_mart_data.describe()
```
Visualizations
Item_Weight distribution
```python
plt.figure(figsize=(6,6))
sns.distplot(big_mart_data['Item_Weight'])
plt.show()
```
Item_Visibility distribution
```python
plt.figure(figsize=(6,6))
sns.distplot(big_mart_data['Item_Visibility'])
plt.show()
```
Item_MRP distribution
```python
plt.figure(figsize=(6,6))
sns.distplot(big_mart_data['Item_MRP'])
plt.show()
```
Item_Outlet_Sales distribution
```python
plt.figure(figsize=(6,6))
sns.distplot(big_mart_data['Item_Outlet_Sales'])
plt.show()
```
Outlet_Establishment_Year count
```python
plt.figure(figsize=(6,6))
sns.countplot(x='Outlet_Establishment_Year', data=big_mart_data)
plt.show()
```
Data Pre-Processing
Standardizing Categorical Variables
```python
big_mart_data.replace({'Item_Fat_Content': {'low fat':'Low Fat', 'LF':'Low Fat', 'reg':'Regular'}}, inplace=True)
```
Label Encoding
```python
encoder = LabelEncoder()
big_mart_data['Item_Identifier'] = encoder.fit_transform(big_mart_data['Item_Identifier'])
big_mart_data['Item_Fat_Content'] = encoder.fit_transform(big_mart_data['Item_Fat_Content'])
big_mart_data['Item_Type'] = encoder.fit_transform(big_mart_data['Item_Type'])
big_mart_data['Outlet_Identifier'] = encoder.fit_transform(big_mart_data['Outlet_Identifier'])
big_mart_data['Outlet_Size'] = encoder.fit_transform(big_mart_data['Outlet_Size'])
big_mart_data['Outlet_Location_Type'] = encoder.fit_transform(big_mart_data['Outlet_Location_Type'])
big_mart_data['Outlet_Type'] = encoder.fit_transform(big_mart_data['Outlet_Type'])
```
Splitting Features and Target
```python
X = big_mart_data.drop(columns='Item_Outlet_Sales', axis=1)
Y = big_mart_data['Item_Outlet_Sales']
```
Splitting the Data into Training and Testing Data
```python
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
```
Machine Learning Model Training
XGBoost Regressor
```python
regressor = XGBRegressor()
regressor.fit(X_train, Y_train)
```
Evaluation
Prediction on Training Data
```python
training_data_prediction = regressor.predict(X_train)
r2_train = metrics.r2_score(Y_train, training_data_prediction)
print('R Squared value = ', r2_train)
```
Prediction on Test Data
```python
test_data_prediction = regressor.predict(X_test)
r2_test = metrics.r2_score(Y_test, test_data_prediction)
print('R Squared value = ', r2_test)
```
Conclusion
The project successfully builds a predictive model to estimate the sales of items in different outlets. The XGBoost Regressor provides a good balance between training and test performance, making it suitable for this regression problem.

Future Work
Explore different regression models and compare their performance.
Perform hyperparameter tuning to optimize the XGBoost model.
Implement feature engineering techniques to improve model accuracy.
