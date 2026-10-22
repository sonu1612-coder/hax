
import numpy as np

a = np.array([10, 20, 30, 40])
print("Array:", a)
print("Mean:", np.mean(a))
print("Sum:", np.sum(a))



import pandas as pd

data = {'Name': ['A', 'B', 'C'], 'Marks': [80, 90, 70]}
df = pd.DataFrame(data)
print(df)
print(df.head())
print(df.describe())



import matplotlib.pyplot as plt

x = [1, 2, 3, 4]
y = [10, 20, 15, 30]

plt.plot(x, y, marker='o')
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Simple Graph")
plt.show()


from sklearn.linear_model import LinearRegression

X = [[1], [2], [3], [4], [5]]
y = [10, 20, 30, 40, 50]

model = LinearRegression()
model.fit(X, y)

print("Prediction:", model.predict([[6]]))
print("Coefficient:", model.coef_)
