# Sales-predictions-
This project is a Sales Prediction system built using Python that analyzes historical sales data to predict future sales trends. It uses data preprocessing and simple machine learning techniques to identify patterns in past sales and estimate future performance. The goal is to help businesses understand demand trends and make better planning 
import pandas as pd

data = {
    "Ads": [1000, 2000, 3000, 4000],
    "Sales": [5000, 9000, 13000, 17000]
}

df = pd.DataFrame(data)

from sklearn.linear_model import LinearRegression

X = df[["Ads"]]
y = df["Sales"]

model = LinearRegression()

model.fit(X, y)

print("Model trained")

prediction = model.predict(
    pd.DataFrame([[6000]], columns=["Ads"])
)

print(prediction)


import matplotlib.pyplot as plt

plt.scatter(df["Ads"], df["Sales"])

plt.plot(X, model.predict(X))

plt.xlabel("Ads")
plt.ylabel("Sales")
plt.title("Sales Prediction")

plt.show()
import matplotlib.pyplot as plt

plt.scatter(df["Ads"], df["Sales"])

plt.plot(X, model.predict(X))

plt.xlabel("Ads")
plt.ylabel("Sales")
plt.title("Sales Prediction")

plt.show()
                  