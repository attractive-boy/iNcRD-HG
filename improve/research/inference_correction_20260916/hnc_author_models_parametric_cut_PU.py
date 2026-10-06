import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors
import copy as cp
import subprocess
import os
from neighborfinder import *
import time
from toBareHPF import toBareHPF
import sys

class HNC_pcut(BaseEstimator, ClassifierMixin):
    
    ## without confidence label
    ## all labeled samples must be in the source/sink set as given

    def __init__(self, list_lambda, k=15, neighboring=True, 
                 distance_metric='euclidean', weight='RBF', epsilon=1, 
                 add_neighbor=False, adjust_lambda=False, pu_learning=False, debug=False):
        self.epsilon = epsilon
        self.list_lambda = sorted(list_lambda)
        self.k = k
        self.neighboring = neighboring
        self.distance_metric = distance_metric
        self.weight = weight
        self.add_neighbor = add_neighbor
        self.adjust_lambda = adjust_lambda
        self.adjust_weight = None # we use the sum of weights in the graph
        self.neighbors_boo = None
        self.neighbors_dist = None
        self.neighbors_sim = None
        self.neighbors_fitted = False
        self.neighbor_method = "knn"
        self.pu_learning = pu_learning # true if we are solving pu learning problem without negative seeds
        self.debug = debug

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        if self.debug:
            if self.pu_learning:
                print("with only positive samples")
            else:
                print("with both positive and negative samples")
        return self
    
    def fit_neighbor(self, neighbor_boo, neighbor_info, mode="dist"):
        # we only need either neighbor_dist or neighbor_sim
        self.neighbors_boo = neighbor_boo
        self.neighbors_fitted = True
        if mode in {"dist", "distance", "distances"}:
            self.neighbors_dist = neighbor_info
        elif mode in {"sim", "similarity", "similarities"}:
            self.neighbors_sim = neighbor_info
        
        return self
    
    def _get_neighbor(self, data_arr, numneigh=10):

        if self.neighbor_method == "knn":
            boo_arr, dist_arr = get_neighbor_knn(data_arr, k=numneigh, metric=self.distance_metric)
        '''
        elif self.neighbor_method == "sparsecomp":
            boo_arr, dist_arr = get_neighbor_sparsecomp(data_arr, k=numneigh, metric=self.distance_metric)
        elif self.neighbor_method == "hnsw":
            boo_arr, dist_arr = get_neighbor_hnsw(data_arr, k=numneigh, metric=self.distance_metric)
        elif self.neighbor_method == "top":
            boo_arr, dist_arr = get_neighbor_top(data_arr, k=numneigh, metric=self.distance_metric)
        '''
        
        return boo_arr, dist_arr
    
    def _add_neighbor(self, data_arr, boo_arr, dist_arr):

        n_train = self.X_train.shape[0]
        unpaired_indices = [n_train + ind for ind in np.where(np.sum(boo_arr[n_train:,:n_train] | boo_arr[:n_train,n_train:].T, axis=1) == 0)[0]]
        
        if len(unpaired_indices) > 0:
            neigh_added = NearestNeighbors(n_neighbors=1, metric=self.distance_metric)
            neigh_added.fit(data_arr[:n_train,:])
            neighbors_added = neigh_added.kneighbors(data_arr[unpaired_indices,:])

            for i, ind in enumerate(unpaired_indices):
                boo_arr[ind,neighbors_added[1][i,0]] = True
                dist_arr[ind,neighbors_added[1][i,0]] = neighbors_added[0][i,0]

        return boo_arr, dist_arr
    
    
    def _get_weight(self, dist, ref_dist=0):
        # input: distance matrix
        # output: edge weight on the graph
        w_metric = self.weight
        epsilon = self.epsilon
        if w_metric == 'RBF':
            w = np.exp(-(dist**2 / (2 * epsilon ** 2)))
        elif w_metric == 'inverse':
            w = 5/dist
        elif w_metric == 'RBF_norm':
            w = np.exp(-((dist-ref_dist)**2 / (2 * epsilon ** 2)))
        elif w_metric == 'new':
            w = 10*ref_dist/np.maximum(dist, 0.001*ref_dist)
        return w

    def _get_distance(self, X):
        distances= squareform(pdist(X, metric=self.distance_metric))
        return distances
    
    def _construct_graph(self, sim, is_neighbor):

        start_timer = time.perf_counter()
        
        n_train, n_samples = self.X_train.shape[0], sim.shape[0]
        n_test = n_samples - n_train 
        
        list_lambda = self.list_lambda
        if self.adjust_lambda:
            list_lambda /= self.adjust_weight
        num_lambda = len(list_lambda)
        
        source_index, sink_index = n_test+1, n_test+2
        lines = ["","n "+str(int(source_index))+" s", "n "+str(int(sink_index))+" t"]

        # first, get the weights of source and sink adjacent arcs

        if self.debug:
            print("step 1: ", time.perf_counter()-start_timer)
        
        pos_booleans = self.y_train == 1
        idx_positives = np.arange(n_train)[pos_booleans]
        if not self.pu_learning:
            idx_negatives = np.arange(n_train)[np.invert(pos_booleans)]
        idx_unlabeled = np.arange(n_train, n_samples)

        if self.debug:
            print("step 1.1: ", time.perf_counter()-start_timer)
        
        source_weights = 0.5*(sim[idx_unlabeled, :][:, idx_positives].sum(axis=1)+sim[:,idx_unlabeled][idx_positives,:].sum(axis=0))
        if not self.pu_learning:
            sink_weights = sim[idx_unlabeled, :][:, idx_negatives].sum(axis=1)
        else:
            sink_weights = np.zeros(source_weights.shape)

        if self.debug:
            print("step 1.2: ", time.perf_counter()-start_timer)
        # adjust sink/source adjacent weights to optimize runtime of the cut algorithm
        if not self.pu_learning:
            min_weights = np.minimum(source_weights, sink_weights)
            source_weights -= min_weights
            sink_weights -= min_weights
        if self.debug:
            print("step 1.3: ", time.perf_counter()-start_timer)
        # compute the weighted degree, which would be multiplied by lambda values later
        deg_weights = sim[idx_unlabeled, :].sum(axis=1)

        if self.debug:
            print("step 2: ", time.perf_counter()-start_timer)

        for i in range(n_test):
            s_source = "a "+str(int(source_index))+" "+str(int(i+1))
            s_sink  = "a "+str(int(i+1))+" "+str(int(sink_index))
            for l in list_lambda:
                if l > 0:
                    s_source += " "+str(source_weights[i]+(l*deg_weights[i]))
                    s_sink += " "+str(sink_weights[i])
                else:
                    s_source += " "+str(source_weights[i])
                    s_sink += " "+str(sink_weights[i]+((-1)*l*deg_weights[i]))
            lines.append(s_source)
            lines.append(s_sink)

        if self.debug:
            print("step 3: ", time.perf_counter()-start_timer)
        
        if self.neighboring:
            for i in range(n_test-1):
                for j in range(i+1,n_test):
                    if is_neighbor[(i+n_train),(j+n_train)] or is_neighbor[(i+n_train),(j+n_train)]:
                        mutual_weight = 0.5*(sim[i+n_train,j+n_train]+sim[j+n_train,i+n_train])
                        s = "a "+str(int(i+1))+" "+str(int(j+1))+" "+str(mutual_weight)
                        lines.append(s)
                        s = "a "+str(int(j+1))+" "+str(int(i+1))+" "+str(mutual_weight)
                        lines.append(s)
        else:
            for i in range(n_test-1):
                for j in range(i+1,n_test):
                    s = "a "+str(int(i+1))+" "+str(int(j+1))+" "+str(0.5*(sim[i+n_train,j+n_train]+sim[j+n_train,i+n_train]))
                    lines.append(s)
                    s = "a "+str(int(j+1))+" "+str(int(i+1))+" "+str(0.5*(sim[i+n_train,j+n_train]+sim[j+n_train,i+n_train]))
                    lines.append(s)
        
        if self.debug:
            print("step 4: ", time.perf_counter()-start_timer)

        numarcs = len(lines) - 3
        lines[0] = "p par-max "+str(int(n_test+2))+" "+str(int(numarcs))+" "+str(int(num_lambda))
        
        with open("./parametric_cut/parametric_cut_input.txt", "w") as file:
            file.writelines("%s\n" % l for l in lines)
            file.close()
        
        if self.debug:
            print("step 5: ", time.perf_counter()-start_timer)
        
        return lines
    
    def _solve_parametric_cut(self, n_test):
        #subprocess.call(["gcc", "./parametric_cut/pseudopar.c"]) need to run this line only if it is not compiled yet
        subprocess.run(['./parametric_cut/compiled_parametric_cut.out'])
        
        file1 = open('./parametric_cut/parametric_cut_output.txt', 'r')
        lines = file1.readlines()
        file1.close()
        
        pred_arr = np.zeros((n_test,len(self.list_lambda)))
        
        for line in lines[:-2]:
            L = line.split()
            pred_arr[int(L[1])-1,int(L[2])-1:] = 1
            
        os.remove('./parametric_cut/parametric_cut_output.txt')
        
        return pred_arr

    def predict(self, X_test):

        X_train = self.X_train
        n_train, n_test = X_train.shape[0], X_test.shape[0]
        n_samples = n_train+n_test
        X_all = np.concatenate((X_train, X_test), axis=0)
        k = self.k
        if k < 1:
            k = int(k*n_samples)
        
        is_neighbor = self.neighbors_boo
        # we only need either distances or similarities
        distances = self.neighbors_dist 
        similarities = self.neighbors_sim # it would be None if it is not provided along with neighbors_boo in fit_neighbors(), and we would have to compute it
        # note that one of similarities[i,j] and similarities[j,i] could be zero, or both of them could be nonzero and equal to one another

        if similarities is None:
            if self.neighboring:
                if not self.neighbors_fitted:
                    is_neighbor, distances = self._get_neighbor(X_all, numneigh=k)
                if self._add_neighbor:
                    is_neighbor, distances = self._add_neighbor(X_all, is_neighbor, distances)
            else:
                is_neighbor = np.full((X_all.shape[0], X_all.shape[0]), True)
                distances = self._get_distance(X_all)
                
            # adjust distances magnitude with respect to the dimension of the data (by dividing np.sqrt(num features))
            similarities = np.zeros(distances.shape)

            if self.weight=="RBF_norm":
                min_distance = distances[is_neighbor].min()
                similarities[is_neighbor] = self._get_weight(distances[is_neighbor]/np.sqrt(X_train.shape[1]), min_distance)
            else:
                similarities[is_neighbor] = self._get_weight(distances[is_neighbor]/np.sqrt(X_train.shape[1]))
        
        #if self.adjust_lambda:
        #    self.adjust_weight = similarities[np.nonzero(similarities)].sum()

        ##### create text input to parametric cut rather than a networkx graph
        
        lines = self._construct_graph(similarities, is_neighbor)

        # and use the parametric cut rather than hpf
        pred_arr = self._solve_parametric_cut(n_test)
               
        return pred_arr

class HNC_pcut_oneclass_csr(BaseEstimator, ClassifierMixin):
    # always take lambda values that are negative (connect nodes to sink with positive values of lambdas)
    # always have labeled samples for the positive class

    def __init__(self, list_lambda, k=15, neighboring=True, 
                 distance_metric='euclidean', weight='RBF', epsilon=1, 
                 add_neighbor=False, adjust_lambda=False, debug=False):
        self.epsilon = epsilon
        self.list_lambda = sorted(list_lambda)
        self.k = k
        self.neighboring = neighboring
        self.distance_metric = distance_metric
        self.weight = weight
        self.add_neighbor = add_neighbor
        self.neighbors_boo = None
        self.neighbors_dist = None
        self.neighbors_sim = None
        self.neighbors_fitted = False
        self.neighbor_method = "knn"
        self.debug = debug

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        '''
        if self.debug:
            if self.pu_learning:
                print("with only positive samples")
            else:
                print("with both positive and negative samples")
        '''
        return self
    
    def fit_neighbor(self, neighbor_csr):
        self.neighbors_csr = neighbor_csr
        return self
    
    def _get_neighbor(self, data_arr, numneigh=10):

        if self.neighbor_method == "knn":
            boo_arr, dist_arr = get_neighbor_knn(data_arr, k=numneigh, metric=self.distance_metric)
        '''
        elif self.neighbor_method == "hnsw":
            boo_arr, dist_arr = get_neighbor_hnsw(data_arr, k=numneigh, metric=self.distance_metric)
        elif self.neighbor_method == "top":
            boo_arr, dist_arr = get_neighbor_top(data_arr, k=numneigh, metric=self.distance_metric)
        elif self.neighbor_method == "sparsecomp":
            boo_arr, dist_arr = get_neighbor_sparsecomp(data_arr, k=numneigh, metric=self.distance_metric)
        '''
        
        return boo_arr, dist_arr
    
    def _add_neighbor(self, data_arr, boo_arr, dist_arr):

        n_train = self.X_train.shape[0]
        unpaired_indices = [n_train + ind for ind in np.where(np.sum(boo_arr[n_train:,:n_train] | boo_arr[:n_train,n_train:].T, axis=1) == 0)[0]]
        
        if len(unpaired_indices) > 0:
            neigh_added = NearestNeighbors(n_neighbors=1, metric=self.distance_metric)
            neigh_added.fit(data_arr[:n_train,:])
            neighbors_added = neigh_added.kneighbors(data_arr[unpaired_indices,:])

            for i, ind in enumerate(unpaired_indices):
                boo_arr[ind,neighbors_added[1][i,0]] = True
                dist_arr[ind,neighbors_added[1][i,0]] = neighbors_added[0][i,0]

        return boo_arr, dist_arr
    
    
    def _get_weight(self, dist, ref_dist=0):
        # input: distance matrix
        # output: edge weight on the graph
        w_metric = self.weight
        epsilon = self.epsilon
        if w_metric == 'RBF':
            w = np.exp(-(dist**2 / (2 * epsilon ** 2)))
        elif w_metric == 'inverse':
            w = 5/dist
        elif w_metric == 'RBF_norm':
            w = np.exp(-((dist-ref_dist)**2 / (2 * epsilon ** 2)))
        elif w_metric == 'new':
            w = 10*ref_dist/np.maximum(dist, 0.001*ref_dist)
        return w

    def _get_distance(self, X):
        distances= squareform(pdist(X, metric=self.distance_metric))
        return distances
    
    def _create_edge_dict(self, csr_arr):

        n_train, n_samples = self.X_train.shape[0], csr_arr.shape[0]
        n_test = n_samples - n_train 
        
        edge_dict = {}
        source_index, sink_index = n_test, n_test+1

        source_weights = 0.5*(csr_arr[n_train:n_samples, :][:, :n_train].sum(axis=1)+csr_arr[:,n_train:n_samples][:n_train,:].sum(axis=0).T)
        deg_weights = csr_arr[n_train:n_samples, :].sum(axis=1)

        for i in range(n_test):
            edge_dict[(source_index, i)] = (source_weights[i,0],0)
            edge_dict[(i, sink_index)] = (0,(-1)*deg_weights[i,0])

        for row, col in zip(*csr_arr.nonzero()):
            if row >= n_train and col >= n_train and row != col:
                if (row-n_train,col-n_train) in edge_dict:
                    edge_dict[(row-n_train, col-n_train)] = (edge_dict[(row-n_train, col-n_train)][0]+0.5*csr_arr[row, col], 0)
                    edge_dict[(col-n_train, row-n_train)] = (edge_dict[(col-n_train, row-n_train)][0]+0.5*csr_arr[row, col], 0)
                else:
                    edge_dict[(row-n_train, col-n_train)] = (0.5*csr_arr[row, col], 0)
                    edge_dict[(col-n_train, row-n_train)] = (0.5*csr_arr[row, col], 0)

        return edge_dict

    def predict(self, X_test):

        X_train = self.X_train
        n_train, n_test = X_train.shape[0], X_test.shape[0]
        n_samples = n_train+n_test
        
        k = self.k
        if k < 1:
            k = int(k*n_samples)
        
        neighbors_csr = self.neighbors_csr

        # use Alex and Roberto's implementation of parametric cut solver
        edges_dict = self._create_edge_dict(neighbors_csr)
        
        breakpoints = toBareHPF(edges_dict, n_test+2, n_test, n_test+1, self.list_lambda)
        pred_arr = np.zeros((n_test,len(self.list_lambda)))
        for i in range(n_test):
            pred_arr[i,breakpoints[i]-1:] = 1
               
        return pred_arr

class HNC_pcut_2class_csr(BaseEstimator, ClassifierMixin):
    # always take lambda values that are positive (connect nodes to sink with positive values of lambdas)
    # always have labeled samples for the positive class

    def __init__(self, list_lambda, k=15, neighboring=True, 
                 distance_metric='euclidean', weight='RBF', epsilon=1, 
                 add_neighbor=False, adjust_lambda=False, debug=False):
        self.epsilon = epsilon
        self.list_lambda = sorted(list_lambda)
        self.k = k
        self.neighboring = neighboring
        self.distance_metric = distance_metric
        self.weight = weight
        self.add_neighbor = add_neighbor
        self.neighbors_boo = None
        self.neighbors_dist = None
        self.neighbors_sim = None
        self.neighbors_fitted = False
        self.neighbor_method = "knn"
        self.debug = debug

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        '''
        if self.debug:
            if self.pu_learning:
                print("with only positive samples")
            else:
                print("with both positive and negative samples")
        '''
        return self
    
    def fit_neighbor(self, neighbor_csr):
        self.neighbors_csr = neighbor_csr
        return self
    
    def _get_neighbor(self, data_arr, numneigh=10):

        if self.neighbor_method == "knn":
            boo_arr, dist_arr = get_neighbor_knn(data_arr, k=numneigh, metric=self.distance_metric)
        '''
        elif self.neighbor_method == "hnsw":
            boo_arr, dist_arr = get_neighbor_hnsw(data_arr, k=numneigh, metric=self.distance_metric)
        elif self.neighbor_method == "top":
            boo_arr, dist_arr = get_neighbor_top(data_arr, k=numneigh, metric=self.distance_metric)
        elif self.neighbor_method == "sparsecomp":
            boo_arr, dist_arr = get_neighbor_sparsecomp(data_arr, k=numneigh, metric=self.distance_metric)
        '''
        
        return boo_arr, dist_arr
    
    def _add_neighbor(self, data_arr, boo_arr, dist_arr):

        n_train = self.X_train.shape[0]
        unpaired_indices = [n_train + ind for ind in np.where(np.sum(boo_arr[n_train:,:n_train] | boo_arr[:n_train,n_train:].T, axis=1) == 0)[0]]
        
        if len(unpaired_indices) > 0:
            neigh_added = NearestNeighbors(n_neighbors=1, metric=self.distance_metric)
            neigh_added.fit(data_arr[:n_train,:])
            neighbors_added = neigh_added.kneighbors(data_arr[unpaired_indices,:])

            for i, ind in enumerate(unpaired_indices):
                boo_arr[ind,neighbors_added[1][i,0]] = True
                dist_arr[ind,neighbors_added[1][i,0]] = neighbors_added[0][i,0]

        return boo_arr, dist_arr
    
    
    def _get_weight(self, dist, ref_dist=0):
        # input: distance matrix
        # output: edge weight on the graph
        w_metric = self.weight
        epsilon = self.epsilon
        if w_metric == 'RBF':
            w = np.exp(-(dist**2 / (2 * epsilon ** 2)))
        elif w_metric == 'inverse':
            w = 5/dist
        elif w_metric == 'RBF_norm':
            w = np.exp(-((dist-ref_dist)**2 / (2 * epsilon ** 2)))
        elif w_metric == 'new':
            w = 10*ref_dist/np.maximum(dist, 0.001*ref_dist)
        return w

    def _get_distance(self, X):
        distances= squareform(pdist(X, metric=self.distance_metric))
        return distances
    
    def _create_edge_dict(self, csr_arr):

        n_train, n_samples = self.X_train.shape[0], csr_arr.shape[0]
        n_pos = np.sum(self.y_train == 1)
        n_neg = n_train - n_pos
        n_test = n_samples - n_train 
        
        edge_dict = {}
        source_index, sink_index = n_test, n_test+1

        source_weights = 0.5*(csr_arr[n_train:n_samples, :][:, :n_pos].sum(axis=1)+csr_arr[:,n_train:n_samples][:n_pos,:].sum(axis=0).T)
        sink_weights = 0.5*(csr_arr[n_train:n_samples, :][:, n_pos:n_train].sum(axis=1)+csr_arr[:,n_train:n_samples][n_pos:n_train,:].sum(axis=0).T)
        deg_weights = csr_arr[n_train:n_samples, :].sum(axis=1)

        for i in range(n_test):
            if source_weights[i,0] > sink_weights[i,0]:
                edge_dict[(source_index, i)] = (source_weights[i,0]-sink_weights[i,0],0)
            else:
                edge_dict[(i, sink_index)] = (sink_weights[i,0]-source_weights[i,0],0)
            
            edge_dict[(source_index, i)] = (0,deg_weights[i,0])

        for row, col in zip(*csr_arr.nonzero()):
            if row >= n_train and col >= n_train and row != col:
                if (row-n_train,col-n_train) in edge_dict:
                    edge_dict[(row-n_train, col-n_train)] = (edge_dict[(row-n_train, col-n_train)][0]+0.5*csr_arr[row, col], 0)
                    edge_dict[(col-n_train, row-n_train)] = (edge_dict[(col-n_train, row-n_train)][0]+0.5*csr_arr[row, col], 0)
                else:
                    edge_dict[(row-n_train, col-n_train)] = (0.5*csr_arr[row, col], 0)
                    edge_dict[(col-n_train, row-n_train)] = (0.5*csr_arr[row, col], 0)

        return edge_dict

    def predict(self, X_test):

        X_train = self.X_train
        n_train, n_test = X_train.shape[0], X_test.shape[0]
        n_samples = n_train+n_test
        
        k = self.k
        if k < 1:
            k = int(k*n_samples)
        
        neighbors_csr = self.neighbors_csr

        # use Alex and Roberto's implementation of parametric cut solver
        edges_dict = self._create_edge_dict(neighbors_csr)
        
        breakpoints = toBareHPF(edges_dict, n_test+2, n_test, n_test+1, self.list_lambda)
        pred_arr = np.zeros((n_test,len(self.list_lambda)))
        for i in range(n_test):
            pred_arr[i,breakpoints[i]-1:] = 1
               
        return pred_arr

## ===============================================================================================
## ===============================================================================================
## ===============================================================================================
