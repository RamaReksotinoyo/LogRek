import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from ml_from_scratch.linear.classification import logistic

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

x, y = load_iris(return_X_y = True, as_frame = True)
y = y.apply(lambda val: 1 if val==2 else -1)

x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.20, random_state=42)

# GENERATE DATA
# This is an OR logic operation
# X1 | X2 | X1 or X2
# 0  | 0  | 0
# 1  | 0  | 1
# 1  | 1  | 1
# 0  | 1  | 1
# data = pd.DataFrame({"x1": [0, 1, 1, 0, 1],
#                      "x2": [0, 0, 1, 1, 1],
#                      "y": [0, 1, 1, 1, 1]})
# X = data[["x1", "x2"]]
# y = data["y"]


# Modeling
clf = logistic.LogisticRegression(penalty='L1')
clf.fit(x_tr, y_tr)
print(clf.predict(x_tr))
print(clf.predict_proba(x_tr))


# Create Decision Boundary
w1, w2, w3, w4 = clf.coef_
w0 = clf.intercept_
m = -w1/w2
c = -w0/w2

# Create prediction
y_pred = clf.predict(x_te)
# data["y_pred"] = y_pred

# Gabungkan x_te dan y_te untuk plotting
data_te = x_te.copy()
data_te['y_pred'] = y_te.values  # Menambahkan kolom y_pred

# PLOT
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))

sns.scatterplot(data=data_te,
                x=data_te.columns[0],  # Ganti dengan nama kolom yang sesuai, misalnya "sepal length"
                y=data_te.columns[1],  # Ganti dengan nama kolom yang sesuai, misalnya "sepal width"
                hue="y_pred",
                s=200,
                ax=ax)

# Plot decision boundary
m = 10  # Ganti dengan kemiringan yang sesuai
c = 0  # Ganti dengan intercept yang sesuai
x_ = np.linspace(-1, 2, 101)
y_ = m*x_ + c
ax.plot(x_, y_, "--r", alpha=0.5, label="decision boundary")

plt.xlim(-1, 2)
plt.ylim(-1, 2)
plt.legend()
plt.grid()
plt.show()

plt.show()