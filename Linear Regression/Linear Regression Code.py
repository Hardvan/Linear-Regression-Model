import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Getting the Data

customers = pd.read_csv("Ecommerce Customers")

print(customers.head())
print()
print(customers.describe())
print()
print(customers.info())
print()


# EDA

sns.jointplot(x = customers["Time on Website"],
             y = customers["Yearly Amount Spent"],
             data = customers)

sns.jointplot(x = "Time on App",
             y = "Yearly Amount Spent",
             data = customers)

sns.jointplot(x = customers["Time on App"],
             y = customers["Length of Membership"],
             data = customers, kind = "hex")

sns.pairplot(customers)

sns.lmplot(x = "Yearly Amount Spent",
          y = "Length of Membership",
          data = customers)


# Training Linear Regression Model

print(customers.columns)
print()

X = customers[['Avg. Session Length', 'Time on App',
       'Time on Website', 'Length of Membership']]
y = customers["Yearly Amount Spent"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

from sklearn.linear_model import LinearRegression

lm = LinearRegression()

lm.fit(X_train, y_train)

print("Coefficients:\n", lm.coef_, sep="")
print()


# Predicting Test Data

predictions = lm.predict(X_test)

# Scatter Plot of Test/Predictions
fig, ax = plt.subplots()

ax.scatter(x = y_test, y = predictions)
plt.xlabel("Y Test")
plt.ylabel("Predicted Y")
plt.title("Test v/s Predictions")


# Evaluating the Model

from sklearn import metrics

mae = metrics.mean_absolute_error(y_test, predictions)
mse = metrics.mean_squared_error(y_test, predictions)
rmse = np.sqrt(metrics.mean_squared_error(y_test, predictions))

print("Model Details")
print("\nMAE: {}\nMSE: {}\nRMSE: {}".format(mae,mse,rmse))
print()

sns.histplot((y_test - predictions), bins = 50)

# Conclusion

print("Coefficients Table")
cdf = pd.DataFrame(lm.coef_, X.columns, columns = ["Coefficient"])
print(cdf)






























