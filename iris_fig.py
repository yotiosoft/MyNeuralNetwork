import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.datasets import load_iris

iris = load_iris()
x1 = iris.data[:, :2]
x2 = iris.data[:, 2:]
y = iris.target
color = ["red", "orange", "green"]
y_color = [color[0] if i == 0 else color[1] if i == 1 else color[2] for i in y]
labels = iris.target_names

fig, axs = plt.subplots(4, 4, figsize=(7, 7))
fig.subplots_adjust(wspace=0.45, hspace=0.5, top=0.92)

# Line 1: Sepal length
axs[0, 0].text(0.5, 0.5, "Sepal length", horizontalalignment="center", verticalalignment="center", transform=axs[0, 0].transAxes)
axs[0, 0].axis("off")

# x1[:,1]のうち、0のデータを取り出す
sc = axs[0, 1].scatter(x1[:, 1], x1[:, 0], c=y_color, cmap=plt.cm.Set1, s=5)
axs[0, 1].set_xlabel("Sepal width")
axs[0, 1].set_ylabel("Sepal length")
axs[0, 1].yaxis.set_major_locator(ticker.MultipleLocator(2))

axs[0, 2].scatter(x2[:, 0], x1[:, 0], c=y_color, cmap=plt.cm.Set1, s=5)
axs[0, 2].set_xlabel("Petal length")
axs[0, 2].set_ylabel("Sepal length")
axs[0, 2].yaxis.set_major_locator(ticker.MultipleLocator(2))

axs[0, 3].scatter(x2[:, 1], x1[:, 0], c=y_color, cmap=plt.cm.Set1, s=5)
axs[0, 3].set_xlabel("Petal width")
axs[0, 3].set_ylabel("Sepal length")
axs[0, 3].yaxis.set_major_locator(ticker.MultipleLocator(2))

# Line 2: Sepal width
axs[1, 0].scatter(x1[:, 0], x1[:, 1], c=y_color, cmap=plt.cm.Set1, s=5)
axs[1, 0].set_xlabel("Sepal length")
axs[1, 0].set_ylabel("Sepal width")
axs[1, 0].yaxis.set_major_locator(ticker.MultipleLocator(2))

axs[1, 1].text(0.5, 0.5, "Sepal width", horizontalalignment="center", verticalalignment="center", transform=axs[1, 1].transAxes)
axs[1, 1].axis("off")

axs[1, 2].scatter(x2[:, 0], x1[:, 1], c=y_color, cmap=plt.cm.Set1, s=5)
axs[1, 2].set_xlabel("Petal length")
axs[1, 2].set_ylabel("Sepal width")
axs[1, 2].yaxis.set_major_locator(ticker.MultipleLocator(2))

axs[1, 3].scatter(x2[:, 1], x1[:, 1], c=y_color, cmap=plt.cm.Set1, s=5)
axs[1, 3].set_xlabel("Petal width")
axs[1, 3].set_ylabel("Sepal width")
axs[1, 3].yaxis.set_major_locator(ticker.MultipleLocator(2))

# Line 3: Petal length
axs[2, 0].scatter(x1[:, 0], x2[:, 0], c=y_color, cmap=plt.cm.Set1, s=5)
axs[2, 0].set_xlabel("Sepal length")
axs[2, 0].set_ylabel("Petal length")
axs[2, 0].yaxis.set_major_locator(ticker.MultipleLocator(2))

axs[2, 1].scatter(x1[:, 1], x2[:, 0], c=y_color, cmap=plt.cm.Set1, s=5)
axs[2, 1].set_xlabel("Sepal width")
axs[2, 1].set_ylabel("Petal length")
axs[2, 1].yaxis.set_major_locator(ticker.MultipleLocator(2))

axs[2, 2].text(0.5, 0.5, "Petal length", horizontalalignment="center", verticalalignment="center", transform=axs[2, 2].transAxes)
axs[2, 2].axis("off")

axs[2, 3].scatter(x2[:, 1], x2[:, 0], c=y_color, cmap=plt.cm.Set1, s=5)
axs[2, 3].set_xlabel("Petal width")
axs[2, 3].set_ylabel("Petal length")
axs[2, 3].yaxis.set_major_locator(ticker.MultipleLocator(2))

# Line 4: Petal width
axs[3, 0].scatter(x1[:, 0], x2[:, 1], c=y_color, cmap=plt.cm.Set1, s=5)
axs[3, 0].set_xlabel("Sepal length")
axs[3, 0].set_ylabel("Petal width")
axs[3, 0].yaxis.set_major_locator(ticker.MultipleLocator(2))

axs[3, 1].scatter(x1[:, 1], x2[:, 1], c=y_color, cmap=plt.cm.Set1, s=5)
axs[3, 1].set_xlabel("Sepal width")
axs[3, 1].set_ylabel("Petal width")
axs[3, 1].yaxis.set_major_locator(ticker.MultipleLocator(2))

axs[3, 2].scatter(x2[:, 0], x2[:, 1], c=y_color, cmap=plt.cm.Set1, s=5)
axs[3, 2].set_xlabel("Petal length")
axs[3, 2].set_ylabel("Petal width")
axs[3, 2].yaxis.set_major_locator(ticker.MultipleLocator(2))

axs[3, 3].text(0.5, 0.5, "Petal width", horizontalalignment="center", verticalalignment="center", transform=axs[3, 3].transAxes)
axs[3, 3].axis("off")

leg = fig.legend(*sc.legend_elements(),
        labels=labels,
        loc='upper center',
        ncol=3,
        bbox_to_anchor=(1.0, 1.38),
        bbox_transform=axs[0, 1].transAxes
        )

num = 0
for leha in leg.legendHandles:
    leha.set_color(color[num])
    num += 1

plt.show()
