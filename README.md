# Sales-predictions-
 project is a Sales Prediction system built using Python that analyzes historical sales data to predict future sales trends. It uses data preprocessing and simple machine learning techniques to identify patterns in past sales and estimate future performance. The goal is to help businesses understand demand trends and make 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Dataset (corrected - equal length)
data = {
    "Ads":   [1000, 1500, 2000, 2500, 3000, 3500],
    "Sales": [16000, 14000, 12000, 10000, 8000, 6000]
}
df = pd.DataFrame(data)

# Input (X) and Output (y)
X = df[["Ads"]]
y = df["Sales"]

# Model training
model = LinearRegression()
model.fit(X, y)

print("Model trained successfully")

# Prediction for new value
prediction = model.predict(pd.DataFrame([[6000]], columns=["Ads"]))
print("Prediction for 6000 Ads:", prediction)

# Graph
plt.scatter(df["Ads"], df["Sales"])
plt.plot(X, model.predict(X))
plt.xlabel("Ads")
plt.ylabel("Sales")
plt.title("Sales Prediction Model")
plt.show()
