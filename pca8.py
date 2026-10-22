
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
print(df.head())
print(df.describe())

X = StandardScaler().fit_transform(df)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

result = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
print("\nDecomposed Values:\n", result.head())
print("\nExplained Variance:", pca.explained_variance_ratio_)
print("\nComponents:\n", pca.components_)

plt.scatter(result['PC1'], result['PC2'], c=iris.target)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA - Iris Dataset")
plt.show()
