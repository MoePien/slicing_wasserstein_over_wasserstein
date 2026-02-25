import numpy as np
from sklearn.neighbors import KNeighborsClassifier

def calc_KNN_acc(data1, target1, data2, target2, k=5):
    """
    Symmetric KNN accuracy:
      - Train KNN on (data2, target2), test on (data1, target1)
      - Train KNN on (data1, target1), test on (data2, target2)
      - Return the average of the two accuracies
    """
    # Train on data2 → predict data1
    knn12 = KNeighborsClassifier(n_neighbors=k)
    knn12.fit(data2, target2)
    acc12 = knn12.score(data1, target1)

    # Train on data1 → predict data2
    knn21 = KNeighborsClassifier(n_neighbors=k)
    knn21.fit(data1, target1)
    acc21 = knn21.score(data2, target2)

    return (acc12 + acc21) / 2.0
