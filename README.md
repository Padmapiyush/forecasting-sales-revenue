# forecasting-sales-revenue
Created predictive models using machine learning algorithms to forecast sales revenue for a retail company.

code in Python for building a predictive model using machine learning algorithms to forecast sales revenue for a retail company:

```python
# import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# load the data
data = pd.read_csv('sales_data.csv')

# preprocess the data
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
data = data.resample('M').sum() # aggregate monthly sales

# feature engineering
data['Year'] = data.index.year
data['Month'] = data.index.month
data['Quarter'] = data.index.quarter

# visualize the data
sns.lineplot(data=data, x='Date', y='Sales')

# split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data.drop('Sales', axis=1), data['Sales'], test_size=0.2, random_state=42)

# train linear regression model
lr = LinearRegression()
lr.fit(X_train, y_train)

# evaluate the model on test set
y_pred_lr = lr.predict(X_test)
print('Linear Regression MAE:', mean_absolute_error(y_test, y_pred_lr))
print('Linear Regression MSE:', mean_squared_error(y_test, y_pred_lr))
print('Linear Regression R2 Score:', r2_score(y_test, y_pred_lr))

# train decision tree regression model
dtr = DecisionTreeRegressor()
dtr.fit(X_train, y_train)

# evaluate the model on test set
y_pred_dtr = dtr.predict(X_test)
print('Decision Tree Regression MAE:', mean_absolute_error(y_test, y_pred_dtr))
print('Decision Tree Regression MSE:', mean_squared_error(y_test, y_pred_dtr))
print('Decision Tree Regression R2 Score:', r2_score(y_test, y_pred_dtr))

# train random forest regression model
rfr = RandomForestRegressor()
rfr.fit(X_train, y_train)

# evaluate the model on test set
y_pred_rfr = rfr.predict(X_test)
print('Random Forest Regression MAE:', mean_absolute_error(y_test, y_pred_rfr))
print('Random Forest Regression MSE:', mean_squared_error(y_test, y_pred_rfr))
print('Random Forest Regression R2 Score:', r2_score(y_test, y_pred_rfr))

# train XGBoost regression model
xgb = XGBRegressor()
xgb.fit(X_train, y_train)

# evaluate the model on test set
y_pred_xgb = xgb.predict(X_test)
print('XGBoost Regression MAE:', mean_absolute_error(y_test, y_pred_xgb))
print('XGBoost Regression MSE:', mean_squared_error(y_test, y_pred_xgb))
print('XGBoost Regression R2 Score:', r2_score(y_test, y_pred_xgb))
```

You can also try other machine learning algorithms, hyperparameter tuning, and feature selection techniques to improve the performance of the predictive model.
