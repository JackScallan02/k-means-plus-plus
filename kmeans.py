import numpy as np
import scipy as sp
import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
iris = load_iris()

X = iris.data
y = iris.target
Z = pd.DataFrame({'x1': X[:, 0]/X[:, 1], 'x2': X[:, 2]/X[:, 3]})


def k_init(X, k):
    """ k-means++: initialization algorithm

    Parameters
    ----------
    X: array, shape(n ,d)
        Input array of n samples and d features

    k: int
        The number of clusters

    Returns
    -------
    init_centers: array (k, d)
        The initialize centers for kmeans++
    """

    init_centers = np.array([[]])
    c1 = X.sample()

    init_centers = np.append(init_centers, c1, 1)

    d = np.zeros((X.shape[0]))

    #Iterate k - 1 times because we want k centers (we already have one)
    for i in range(k - 1):

        #Iterate through each data point
        for j in range(X.shape[0]):

            #Compute this data point's distance to every center
            distance_array = np.zeros(np.shape(init_centers))
            for c in range(len(init_centers)):
                distance_array[c] = np.linalg.norm(X.iloc[j] - init_centers[c])


            d[j] = np.min(distance_array)

        #Get probability of every point to be the next center
        distanceSum = np.sum(d)
        p = np.zeros((X.shape[0]))
        for j in range(len(d)):
            p[j] = d[j]/distanceSum


        sampled_row = X.sample(n=1, weights=p)

        init_centers = np.append(init_centers, [sampled_row.iloc[0]], 0)


    return init_centers




def k_means_pp(X, k, max_iter):
    """ k-means++ clustering algorithm

    step 1: call k_init() to initialize the centers
    step 2: iteratively refine the assignments

    Parameters
    ----------
    X: array, shape(n ,d)
        Input array of n samples and d features

    k: int
        The number of clusters

    max_iter: int
        Maximum number of iteration

    Returns
    -------
    final_centers: array, shape (k, d)
        The final cluster centers
    """
    centers = k_init(X, k)

    #Iterate max_iter times
    for i in range(max_iter):

        A = assign_data2clusters(X, centers)
        for i in range(len(centers)):

            sum_x1 = 0
            sum_x2 = 0
            for j, val in enumerate(A.T[i]):
                if val == 1:
                    sum_x1 += X.iloc[j]['x1']
                    sum_x2 += X.iloc[j]['x2']

            col_num = A[:, i].sum()

            if col_num != 0:
                centers[i][0] = sum_x1/col_num
                centers[i][1] = sum_x2/col_num


    final_centers = centers
    return final_centers



def assign_data2clusters(X, C):
        """ Assignments of data to the clusters
        Parameters
        ----------
        X: array, shape(n ,d)
            Input array of n samples and d features

        C: array, shape(k ,d)
            The final cluster centers

        Returns
        -------
        data_map: array, shape(n, k)
            The binary matrix A which shows the assignments of data points (X) to
            the input centers (C).
        """
        A = np.zeros((X.shape[0], len(C)))

        #Iterate through every data point
        for index, row in X.iterrows():

            minCenter = C[0]
            minDistance = np.linalg.norm(row.values - C[0])
            minCenterIndex = 0

            #Iterate through each center
            for i, center in enumerate(C):
                curDistance = np.linalg.norm(row.values - center)
                if curDistance < minDistance:
                    minCenter = center
                    minDistance = curDistance
                    minCenterIndex = i


            A[index][minCenterIndex] = 1

        return A


def compute_objective(X, C):
    """ Compute the clustering objective for X and C
    Parameters
    ----------
    X: array, shape(n ,d)
        Input array of n samples and d features

    C: array, shape(k ,d)
        The final cluster centers

    Returns
    -------
    accuracy: float
        The objective for the given assigments
    """
    accuracy = 0
    A = assign_data2clusters(X, C)
    #Want to get the accuracy for each center
    accuracy_by_centers = np.zeros((len(C)))
    for j in range(len(C)):
        for i, val in enumerate(A.T[j]):
            if val == 1:
                accuracy += np.linalg.norm(X.iloc[i] - C[j])

    return accuracy
