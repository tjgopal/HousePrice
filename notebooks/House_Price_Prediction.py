#!/usr/bin/env python
# coding: utf-8

# In[7]:


# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = r"C:\Users\JAYENDRA GOPAL\OneDrive\Documents\Downloads\housing.csv"
housing_data = pd.read_csv(file_path, delim_whitespace=True)

# Renaming columns for readability
housing_data.columns = [
    'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 
    'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV'
]

# Checking for missing values
print(housing_data.isnull().sum())




# In[8]:


# Explore the data
housing_data.describe()

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(housing_data.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Scatter plot between RM (average rooms) and MEDV (house price)
plt.scatter(housing_data['RM'], housing_data['MEDV'])
plt.xlabel("Average Rooms per Dwelling (RM)")
plt.ylabel("Median Value of Homes (MEDV)")
plt.title("RM vs MEDV")
plt.show()


# In[9]:


# Splitting the data into features (X) and target (y)
X = housing_data.drop('MEDV', axis=1)
y = housing_data['MEDV']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training (Linear Regression)
model = LinearRegression()
model.fit(X_train, y_train)


# In[10]:


#Model Evaluation
# Predicting on the test set
y_pred = model.predict(X_test)

# Evaluation Metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")


# In[12]:


# Scatter plot of Actual vs Predicted values
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.show()

# Residual plot (Seaborn requires the data to be passed as x and y keywords)
sns.residplot(x=y_test, y=y_pred, lowess=True, color="g")
plt.xlabel("Actual Prices")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()


# In[ ]:




