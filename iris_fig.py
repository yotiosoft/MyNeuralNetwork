import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris = load_iris()
x1 = iris.data[:, :2]
x2 = iris.data[:, 2:]
y = iris.target

fig, axs = plt.subplots(4, 4, figsize=(7, 7))

# Line 1: Sepal length
axs[0, 1].scatter(x1[:, 1], x1[:, 0], c=y, cmap=plt.cm.Set1, s=5)
axs[0, 1].set_xlabel("Sepal width")
axs[0, 1].set_ylabel("Sepal length")

axs[0, 2].scatter(x2[:, 0], x1[:, 0], c=y, cmap=plt.cm.Set1, s=5)
axs[0, 2].set_xlabel("Petal length")
axs[0, 2].set_ylabel("Sepal length")

axs[0, 3].scatter(x2[:, 1], x1[:, 0], c=y, cmap=plt.cm.Set1, s=5)
axs[0, 3].set_xlabel("Petal width")
axs[0, 3].set_ylabel("Sepal length")

# Line 2: Sepal width
axs[1, 0].scatter(x1[:, 0], x1[:, 1], c=y, cmap=plt.cm.Set1, s=5)
axs[1, 0].set_xlabel("Sepal length")
axs[1, 0].set_ylabel("Sepal width")

axs[1, 2].scatter(x2[:, 0], x1[:, 1], c=y, cmap=plt.cm.Set1, s=5)
axs[1, 2].set_title("Sepal width x Petal length")
axs[1, 2].set_xlabel("Petal length")
axs[1, 2].set_ylabel("Sepal width")

axs[1, 3].scatter(x2[:, 1], x1[:, 1], c=y, cmap=plt.cm.Set1, s=5)
axs[1, 3].set_xlabel("Petal width")
axs[1, 3].set_ylabel("Sepal width")

# Line 3: Petal length
axs[2, 0].scatter(x1[:, 0], x2[:, 0], c=y, cmap=plt.cm.Set1, s=5)
axs[2, 0].set_xlabel("Sepal length")
axs[2, 0].set_ylabel("Petal length")

axs[2, 1].scatter(x1[:, 1], x2[:, 0], c=y, cmap=plt.cm.Set1, s=5)
axs[2, 1].set_xlabel("Sepal width")
axs[2, 1].set_ylabel("Petal length")

axs[2, 3].scatter(x2[:, 1], x2[:, 0], c=y, cmap=plt.cm.Set1, s=5)
axs[2, 3].set_xlabel("Petal width")
axs[2, 3].set_ylabel("Petal length")

# Line 4: Petal width
axs[3, 0].scatter(x1[:, 0], x2[:, 1], c=y, cmap=plt.cm.Set1, s=5)
axs[3, 0].set_xlabel("Sepal length")
axs[3, 0].set_ylabel("Petal width")

axs[3, 1].scatter(x1[:, 1], x2[:, 1], c=y, cmap=plt.cm.Set1, s=5)
axs[3, 1].set_xlabel("Sepal width")
axs[3, 1].set_ylabel("Petal width")

axs[3, 2].scatter(x2[:, 0], x2[:, 1], c=y, cmap=plt.cm.Set1, s=5)
axs[3, 2].set_xlabel("Petal length")
axs[3, 2].set_ylabel("Petal width")

fig.tight_layout()

plt.show()
