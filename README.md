# House_Price_Prediction
It predicts the house price following a popular ML algorithm i.e. Linear Regression
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

df = pd.read_csv('/content/data.csv')

df.head()


df.info()

df.describe()

df.shape

df.isnull().sum()

corr_matrix = df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

sns.histplot(data=df, x="condition", kde=True)
plt.grid()
plt.show()

df=df.drop_duplicates()

x = df[['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated']]
y = df['price']


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size =0.2, random_state=42)

model = LinearRegression()

model.fit(x_train,y_train)

pred = model.predict(x_test)


mse = mean_squared_error(y_test, pred)
rmse = np.sqrt(mse)
print('Root Mean Squared Error:', rmse)

from sklearn.metrics import mean_squared_error

y_pred = model.predict(x_test)

mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

plt.scatter(x_test['sqft_living'], y_test, color='blue', label='Actual')
plt.plot(x_test['sqft_living'], y_pred, color='red', label='Predicted')
plt.xlabel('Sqft Living')
plt.ylabel('Price')
plt.legend()
plt.show()

