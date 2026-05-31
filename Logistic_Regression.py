%pip install seaborn
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import recall_score, f1_score

df = pd.read_csv('CO2 Emissions_Canada.csv')
print(df.columns)

#Creating a column of 1's and 0's
threshold = 245 #g/km
df['Legal Limit'] = df['CO2 Emissions(g/km)'].apply(lambda x: 1 if x <= threshold else 0)
df.to_csv("CO2 Emissions_Canada.csv", index=False)
print(df.columns)

df.drop(['Make', 'Model', 'Vehicle Class', 'Transmission'], axis=1, errors='ignore', inplace=True)
df.head()

df_encoded = pd.get_dummies(df, columns=['Fuel Type'], prefix='', prefix_sep='', drop_first=True)
df_encoded = df_encoded.astype(int)
df_encoded.head()

print(df_encoded.columns)
X = df_encoded[['Engine Size(L)', 'Cylinders', 'Fuel Consumption City (L/100 km)',
       'Fuel Consumption Hwy (L/100 km)', 'Fuel Consumption Comb (L/100 km)',
       'Fuel Consumption Comb (mpg)','CO2 Emissions(g/km)', 'E', 'N', 'X',
       'Z']]
y = df_encoded['Legal Limit'].values
print(df_encoded.columns)

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)
X_train = StandardScaler().fit_transform(X_train)
X_test = StandardScaler().fit_transform(X_test)

from sklearn.linear_model import LogisticRegression
logisticregression = LogisticRegression()
logisticregression.fit(X_train, Y_train)

Y_pred = logisticregression.predict(X_test)

from sklearn.metrics import accuracy_score, precision_score
accuracy = accuracy_score(Y_test, Y_pred)
precision = precision_score(Y_test, Y_pred)
recall = recall_score(Y_test, Y_pred)
f1 = f1_score(Y_test, Y_pred)

Y_train_pred = logisticregression.predict(X_train)

print("Training Metrics:")
print("Accuracy:", accuracy_score(Y_train, Y_train_pred))
print("Precision:", precision_score(Y_train, Y_train_pred))
print("Recall:", recall_score(Y_train, Y_train_pred))
print("F1:", f1_score(Y_train, Y_train_pred))
print("\nTest Metrics:")
print("Accuracy:", accuracy_score(Y_test, Y_pred))
print("Precision:", precision_score(Y_test, Y_pred))
print("Recall:", recall_score(Y_test, Y_pred))
print("F1:", f1_score(Y_test, Y_pred))






