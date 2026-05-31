import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import recall_score, f1_score

df = pd.read_csv('CO2 Emissions_Canada.csv')

df.drop(['Make', 'Model', 'Vehicle Class', 'Transmission'], axis=1, errors='ignore', inplace=True)

threshold = 245 #g/km
df['Legal Limit'] = df['CO2 Emissions(g/km)'].apply(lambda x: 1 if x <= threshold else 0)
df.to_csv("CO2 Emissions_Canada.csv", index=False)
print(df.columns)

df_encoded = pd.get_dummies(df, columns=['Fuel Type'], prefix='', prefix_sep='', drop_first=True)
df_encoded = df_encoded.astype(int)

X = df_encoded[['Engine Size(L)', 'Cylinders', 'Fuel Consumption City (L/100 km)',
       'Fuel Consumption Hwy (L/100 km)', 'Fuel Consumption Comb (L/100 km)',
       'Fuel Consumption Comb (mpg)', 'E', 'N', 'X',
       'Z']]
y = df_encoded['Legal Limit'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
clf = DecisionTreeClassifier(max_depth=4, random_state=42)
clf.fit(X_train, y_train)
# Make predictions on test set
y_pred = clf.predict(X_test)

# Visualize the decision tree
plt.figure(figsize=(20, 10))
plot_tree(
    clf,
    feature_names=X.columns,
    class_names=['Above Threshold', 'Below or Equal Threshold'],
    filled=True,
    rounded=True,
    fontsize=10
)
plt.title('Decision Tree Classifier - CO2 Emissions Classification')
plt.show()

from sklearn.metrics import accuracy_score, precision_score
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)


y_train_pred = clf.predict(X_train)

accuracy_train = accuracy_score(y_train, y_train_pred)
precision_train = precision_score(y_train, y_train_pred)
recall_train = recall_score(y_train, y_train_pred)
f1_train = f1_score(y_train, y_train_pred)

print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
print(f"y_train: {y_train.shape}, y_test: {y_test.shape}")
print("Training Metrics:")
print("Accuracy:", accuracy_score(y_train, y_train_pred))
print("Precision:", precision_score(y_train, y_train_pred))
print("Recall:", recall_score(y_train, y_train_pred))
print("F1:", f1_score(y_train, y_train_pred))
print("\nTest Metrics:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1:", f1_score(y_test, y_pred))






