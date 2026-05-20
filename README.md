# Sales-predictions-
 project is a Sales Prediction system built using Python that analyzes historical sales data to predict future sales trends. It uses data preprocessing and simple machine learning techniques to identify patterns in past sales and estimate future performance. The goal is to help businesses understand demand trends and make better
 
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

    data = {
    "Ads":   [1000, 1500, 2000, 2500, 3000, 3500, 4000],
    "Sales": [16000, 14000, 12000, 10000, 8000, 6000, 4000]
}
df = pd.DataFrame(data)

X = df[["Ads"]]
y = df["Sales"]

model = LinearRegression()
model.fit(X, y)

print("Model trained")

prediction = model.predict(pd.DataFrame([[6000]], columns=["Ads"]))
print(prediction)

plt.scatter(df["Ads"], df["Sales"])
plt.plot(X, model.predict(X))
plt.xlabel("Ads")
plt.ylabel("Sales")
plt.title("Sales Prediction")
plt.show()
