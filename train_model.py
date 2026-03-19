import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
data= pd.read_csv("Housing.csv")




# data = {
#     "Bedrooms": [3,4,2,5,4,3,5,2,4,3],
#     "Size": [1500,2000,850,2500,1800,1300,2400,900,1750,1400],
#     "Age": [10,5,20,2,7,15,3,25,8,12],
#     "Price": [300000,400000,200000,500000,370000,280000,480000,220000,350000,310000]
# }

df = pd.DataFrame(data)
# print("Dataset:\n", df)



df = df.drop("Address", axis=1)


X = df.drop("Price", axis=1)
y = df["Price"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)



y_pred = model.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# print("\nPredicted:", y_pred)
# print("Actual:", y_test.values)
print("MSE:", mse)
print("R2:", r2)
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved successfully")
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)


plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, label="Actual", color="blue")
plt.plot([y_test.min(), y_test.max()],
         [y_pred.min(), y_pred.max()],
         color="red", linewidth=2)
# plt.plot(X_test, y_pred, color="red", label="Predicted")
plt.xlabel("Size (sq ft)")
plt.ylabel("Price ($)")
plt.title("Linear Regression Line")
plt.legend()
plt.show()

