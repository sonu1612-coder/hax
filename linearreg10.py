import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Dataset
data = {
    'Hours': [1, 2, 3, 4, 5],
    'Marks': [20, 30, 40, 50, 60]
}

df = pd.DataFrame(data)

X = df[['Hours']]
y = df['Marks']

# Train model
model = LinearRegression()
model.fit(X, y)

# Prediction
pred = model.predict(X)

print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)
print("Predicted marks for 6 hours:", model.predict([[6]])[0])

# Plot graph
plt.scatter(X, y, label="Actual Data")
plt.plot(X, pred, color='red', label="Regression Line")
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Linear Regression")
plt.legend()
plt.grid(True)
plt.show()