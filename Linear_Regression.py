%pip install seaborn
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


df = pd.read_csv("CO2 Emissions_Canada.csv")

df.head()

df.describe()

df.drop(['Make', 'Model', 'Vehicle Class', 'Transmission'], axis=1, errors='ignore', inplace=True)
df.head()

df_encoded = pd.get_dummies(df, columns=['Fuel Type'], prefix='', prefix_sep='', drop_first=True)
df_encoded.head()

df_encoded = df_encoded.astype(int)
df_encoded.head()

# X#  # a# n# d#  # Y#  # d# a# t# a

df_d = df_encoded.drop(['Legal Limit'], axis=1, errors='ignore', inplace=True)
numeric_df = df_encoded.select_dtypes(include=['number'])
# Compute the correlation matrix
correlation_matrix = numeric_df.corr()
# Set up the matplotlib figure
plt.figure(figsize=(10, 8))
# Create a heatmap using Seaborn
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
# Set plot title
plt.title('Correlation Heatmap')
# Display the heatmap
plt.show()

print(df_encoded.columns)
x = df_encoded[['Engine Size(L)', 'Cylinders', 'Fuel Consumption City (L/100 km)',
       'Fuel Consumption Hwy (L/100 km)', 'Fuel Consumption Comb (L/100 km)',
       'Fuel Consumption Comb (mpg)', 'E', 'N', 'X',
       'Z']]
Y = df_encoded['CO2 Emissions(g/km)']

# T# e# s# t#  # a# n# d#  # t# r# a# i# n#  # d# a# t# a#  # s# p# l# i# t

x_train, x_test, Y_train, Y_test = train_test_split(x, Y, test_size=0.20, random_state=9)

x_regression = LinearRegression().fit(x_train, Y_train)
print(x_regression.intercept_)
print(x_regression.coef_)

Y_prediction = x_regression.predict(x_train)

# P# l# o# t

Y_pred_train = x_regression.predict(x_train)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
mae_train = mean_absolute_error(Y_train, Y_pred_train)
mse_train = mean_squared_error(Y_train, Y_pred_train)
r2_train = r2_score(Y_train, Y_pred_train)
# Print the results
print("Training Metrics:")
print(f'Mean Absolute Error for Test data: {mae_train}')
print(f'Mean Squared Error for Test data: {mse_train}')
print(f'R-squared for Test data: {r2_train}')

Y_prediction = x_regression.predict(x_test)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
mae_test = mean_absolute_error(Y_test, Y_prediction)
mse_test = mean_squared_error(Y_test, Y_prediction)
r2_test = r2_score(Y_test, Y_prediction)
# Print the results
print("Testing Metrics:")
print(f'Mean Absolute Error for Train data: {mae_test}')
print(f'Mean Squared Error for Train data: {mse_test}')
print(f'R-squared for Train data: {r2_test}')



