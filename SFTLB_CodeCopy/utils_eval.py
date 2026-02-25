import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
import sklearn

def knn_accuracy_from_distance_matrix(D, labels, k=3, n_splits=10, test_size=0.3, random_state=42):
    """
    Evaluate k-NN classification accuracy using a precomputed distance matrix.

    Parameters
    ----------
    D : (N, N) ndarray
        Precomputed pairwise distance matrix.
    labels : (N,) iterable
        Class labels for each data point.
    k : int, optional
        Number of neighbors for k-NN.
    n_splits : int, optional
        Number of random train/test splits.
    test_size : float, optional
        Fraction of data to use for testing.
    random_state : int or None
        Seed for reproducibility.

    Returns
    -------
    mean_acc : float
        Mean accuracy over splits.
    std_acc : float
        Standard deviation of accuracy over splits.
    """

    labels = np.asarray(labels)
    assert D.shape[0] == D.shape[1] == labels.shape[0], "Mismatched dimensions."

    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
    accuracies = []

    for train_idx, test_idx in sss.split(D, labels):
        D_train = D[np.ix_(train_idx, train_idx)]
        D_test = D[np.ix_(test_idx, train_idx)]

        y_train = labels[train_idx]
        y_test = labels[test_idx]

        clf = KNeighborsClassifier(n_neighbors=k, metric='precomputed')
        clf.fit(D_train, y_train)
        y_pred = clf.predict(D_test)
        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)

    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)

    print(f"Mean accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    return mean_acc, std_acc

