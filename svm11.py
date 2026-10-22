import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
df['flower_name'] = df.target.apply(lambda x: iris.target_names[x])

print(df.head())

df0 = df[df.target == 0]
df1 = df[df.target == 1]
df2 = df[df.target == 2]

plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.scatter(df0['sepal length (cm)'], df0['sepal width (cm)'], color='green', marker='+')
plt.scatter(df1['sepal length (cm)'], df1['sepal width (cm)'], color='blue', marker='.')
plt.scatter(df2['sepal length (cm)'], df2['sepal width (cm)'], color='red', marker='x')
plt.show()

plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.scatter(df0['petal length (cm)'], df0['petal width (cm)'], color='green', marker='+')
plt.scatter(df1['petal length (cm)'], df1['petal width (cm)'], color='blue', marker='.')
plt.scatter(df2['petal length (cm)'], df2['petal width (cm)'], color='red', marker='x')
plt.show
X = df.drop(['target', 'flower_name'], axis='columns')
y = df.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train SVM model
model = SVC()
model.fit(X_train, y_train)

# Check accuracy
print("Accuracy:", model.score(X_test, y_test))

# Predict flower species
predictions = model.predict([
    [4.8, 3.0, 1.5, 0.3],
    [6.5, 3.0, 5.5, 2.0],
    [5.5, 2.8, 4.2, 1.3]
])

for prediction in predictions:
    print("Predicted Flower:", iris.target_names[prediction])

# SVM with C = 100
model_C = SVC(C=100)
model_C.fit(X_train, y_train)
print("Accuracy (C=100):", model_C.score(X_test, y_test))

# SVM with C = 10
model_C = SVC(C=10)
model_C.fit(X_train, y_train)
print("Accuracy (C=10):", model_C.score(X_test, y_test))

# SVM with Linear Kernel
model_linear = SVC(kernel='linear')
model_linear.fit(X_train, y_train)
print("Accuracy (Linear Kernel):", model_linear.score(X_test, y_test))