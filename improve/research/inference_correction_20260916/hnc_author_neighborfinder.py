import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

def get_similarity(dist, epsilon):
    w = np.exp(-(dist**2 / (2 * epsilon ** 2)))
    return w

def get_neighbor_knn(data, another_data=None, k=10, metric='euclidean'):

    # note on the output: boo_arr is not necessarily symmetric
    # if boo_arr[i,j] is True, it means j is a neighbor of i but not necessarily vice versa

    if another_data is not None:
        boo_arr = np.full((another_data.shape[0], data.shape[0]), False)
        dist_arr = np.zeros((another_data.shape[0], data.shape[0]))
        neigh = NearestNeighbors(n_neighbors=k, metric=metric)
    else:
        boo_arr = np.full((data.shape[0], data.shape[0]), False)
        dist_arr = np.zeros((data.shape[0], data.shape[0]))
        neigh = NearestNeighbors(n_neighbors=1+k, metric=metric)

    neigh.fit(data)

    if another_data is not None:
        neighbors = neigh.kneighbors(another_data)
        for i in range(another_data.shape[0]):
            for ind, j in enumerate(neighbors[1][i,:]):
                boo_arr[i][j] = True
                dist_arr[i][j] = neighbors[0][i,ind]
    else:
        neighbors = neigh.kneighbors(data)
        for i in range(data.shape[0]):
            for ind, j in enumerate(neighbors[1][i,:]):
                if i == j:
                    continue
                boo_arr[i][j] = True
                dist_arr[i][j] = neighbors[0][i,ind]

    return boo_arr, dist_arr

def get_neighbor_csr(data, another_data=None, k=10, metric='euclidean', weight='exp', epsilon=0.75):

    '''
    data : np.array
    another_data : np.array, None if we find knn for data itself, otherwise we find knn among another_data for samples in data
    '''

    # note on the output: boo_arr is not necessarily symmetric
    # if boo_arr[i,j] is True, it means j is one of the k nearest neighbors of i, but not necessarily vice versa

    if another_data is not None:
        boo_arr = np.full((another_data.shape[0], data.shape[0]), False)
        dist_arr = np.zeros((another_data.shape[0], data.shape[0]))
        neigh = NearestNeighbors(n_neighbors=k, metric=metric)
    else:
        neigh = NearestNeighbors(n_neighbors=k+1, metric=metric)

    neigh.fit(data)

    if another_data is not None:
        neighbors = neigh.kneighbors(another_data) # return neigh_dist, neigh_index
        for i in range(another_data.shape[0]):
            boo_arr[i][neighbors[1][i,:]] = True
            dist_arr[i][neighbors[1][i,:]] = neighbors[0][i,:]
        
        col = neighbors[1].flatten()
        row = np.repeat(np.arange(data.shape[0]), k)
        dist_data = neighbors[0].flatten()
        csr = csr_matrix((dist_data, (row, col)), shape=(data.shape[0], another_data.shape[0]))
            
    else:
        neighbors = neigh.kneighbors(data)
        col = neighbors[1].flatten()
        row = np.repeat(np.arange(data.shape[0]), k+1)
        dist_data = neighbors[0].flatten()
        if weight == 'exp':
            dist_data = get_similarity(dist_data/np.sqrt(data.shape[1]), epsilon)
        csr = csr_matrix((dist_data, (row, col)), shape=(data.shape[0], data.shape[0]))

    return csr