import numpy as np
import matplotlib.pyplot as plt
import os

def knn(X, y, z, k=1):
    d = np.sum((X - z) ** 2, axis=1)
    idx = np.argsort(d)[:k]
    cls, vote = np.unique(y[idx], return_counts=True)
    return cls[np.argmax(vote)]

y = []
X = []
for f in os.listdir():
    if not os.path.isfile(f) and not f.startswith("."):
        for i in os.listdir(f):
            if i.endswith(".jpg"):
                x = plt.imread(f+"/"+i)
                X.append(x.flatten())
                y.append(f)
X = np.array(X)
print(X.shape)
print(y)