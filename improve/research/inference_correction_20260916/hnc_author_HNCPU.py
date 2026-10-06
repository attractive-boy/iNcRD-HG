import numpy as np
from readData import readData, data_prep, data_prep_other, news_data_prep, mnist_data_prep
from neighborfinder import get_neighbor_csr
from models_parametric_cut_PU import HNC_pcut_oneclass_csr, HNC_pcut_2class_csr
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from trees import PUExtraTrees
import os
import sys
import argparse
import time

def process_args(arguments):
    parser = argparse.ArgumentParser(
        description='HNC-PU Learning implementation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--dataset', '-d', default='other', type=str,
                        help='The dataset name')
    parser.add_argument('--labeled', '-l', default=0.6, type=float,
                        help='# of labeled data')
    parser.add_argument('--labeledclass', '-lc', default=1, type=int,
                        help='class of labeled data')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--seed', '-S',default=0, type=int,
                        help='Random seed for splitting dataset into training and test set')
    parser.add_argument('--negativeseed', '-M', default=0, type=int,
                        help='Random seed for sampling reliable negatives')
    parser.add_argument('--prior', '-pr', default=None, type=float,
                        help='prior of positive fraction')
    parser.add_argument('--priorshift', '-P', default=1., type=float,
                        help='(mis)specification of prior')

    parser.add_argument('--mode_numreliable', '-mode_nr', default=1, type=int,
                        help='how to select the number of reliable negative, taking 0 for equal to numlabeled, taking 1 if want to scale according to prior')
    
    parser.add_argument('--select_reliable_method', '-sel_method', default='ranking', type=str,
                        help='how to select reliable negatives')

    parser.add_argument('--klarge', '-kl', default=15, type=int,
                        help='number of nearest neighbors in knn sparsification for small data')
    parser.add_argument('--ksmall', '-ks', default=5, type=int,
                        help='number of nearest neighbors in knn sparsification for large data')
    parser.add_argument('--epslarge', '-el', default=0.75, type=float,
                        help='large epsilon for small data')
    parser.add_argument('--epssmall', '-es', default=0.25, type=float,
                        help='small epsilon for large data')
    parser.add_argument('--numlambda', '-nl', default=501, type=int,
                        help='number of values of tradeoff lambda to be used')
    parser.add_argument('--incr_lambda', '-il', default=0.001, type=float,
                        help='increment of values of lambda in the list, starting from 0')
    
    parser.add_argument('--filename', '-fn', default="summary.txt", type=str,
                        help='name of text file to save results')
    
    parser.add_argument('--featureweights', '-fi', default=True, action=argparse.BooleanOptionalAction)
    
    args = parser.parse_args(arguments)

    if args.negativeseed is None:
        args.negativeseed = args.seed+1996
    
    return args

def negative_selector(pred_arr, num_selected=10, method="random", random_seed=27):
    # get the indices in each column of example_diff where the entries are 1
    diff_arr = np.diff(pred_arr, axis=1)
    np.random.seed(random_seed)
    most_neg = np.concatenate([np.random.permutation(np.where(diff_arr[:,i] == 1)[0]) for i in range(diff_arr.shape[1])][::-1])
    if method == "ranking":
        # select the top num_selected indices
        return most_neg[:num_selected], most_neg
    elif method == "alternate":
        # select even entries of most_neg
        return most_neg[:2*num_selected][1::2], most_neg
    elif method == "random":
        # select fist several entries with higher chance, last entries with lower chance
        num_consider = min(num_selected*4, len(most_neg))
        selected = np.random.choice(most_neg[:num_consider], min(num_selected, len(most_neg)), replace=False, p=np.arange(num_consider,0,-1)/(num_consider*(num_consider+1)/2))
        return selected, most_neg

def tpr(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tp/(tp+fn)

def tnr(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn/(tn+fp)

def pu_hnc_csr(data_X, data_y, data_name, labeled_size, prior=None, num_reliable_negative=50, k_sparsify=None, feature_importance=False, negative_selector_method=["random"], metrics=['acc','f1','tpr','tnr'], select_neg_seed=27, args_HNC=None, fi=None):

    # compute feature importance if needed
    # and scale the data using the feature importance
    if feature_importance:
        if fi is None:
            g = PUExtraTrees()
            g.fit(P=data_X[:labeled_size,:], U=data_X, pi = prior)
            fi = g.feature_importances()
        
        data_X = data_X*np.sqrt(data_X.shape[1])*fi
        print("feature importance applied")
    
    if prior is None:
        prior = data_y.mean()
    
    # split data into labeled and unlabeled
    X_train, X_test = data_X[:labeled_size,:], data_X[labeled_size:,:]
    y_train, y_test = data_y[:labeled_size], data_y[labeled_size:]
    
    scorers = {'acc': accuracy_score, 'f1': f1_score, 'tpr': tpr, 'tnr': tnr}

    num_selectors = 1

    if k_sparsify is None:
        k_sparsify = args_HNC['k']
    else:
        args_HNC['k'] = k_sparsify
    
    # sparsify the graph (select pairs to be considered using knn sparsification)
    neighbors_csr = get_neighbor_csr(data_X, k=k_sparsify, epsilon=args_HNC['epsilon'])

    # solve the problem using only positive labeled samples
    # want lambda to be negative for the first step
    if args_HNC['list_lambda'][1] > 0:
        args_HNC['list_lambda'] = sorted([(-1)*lamb for lamb in args_HNC['list_lambda']])

    hnc = HNC_pcut_oneclass_csr(**args_HNC)
    hnc.fit(X_train, y_train)
    hnc.fit_neighbor(neighbors_csr)

    y_pred = hnc.predict(X_test)

    predictions = np.zeros((y_pred.shape[0], 2*y_pred.shape[1]))
    positive_percentage = np.zeros((num_selectors, 2*y_pred.shape[1]))
    if data_name != "other":
        test_scores = np.zeros((num_selectors, len(metrics), 2*y_pred.shape[1]))
        for i in range(y_pred.shape[1]):
            for j, metric in enumerate(metrics):
                test_scores[:,j,i] = scorers[metric](y_test, y_pred[:,i])
    for i in range(y_pred.shape[1]):
        positive_percentage[:,i] = (y_pred[:,i].sum()+labeled_size)/data_X.shape[0]
    predictions[:,:y_pred.shape[1]] = y_pred
    
    # now solve the problem using positive and reliable negative labeled samples
    # (change to positive lambdas)
    if args_HNC['list_lambda'][1] < 0:
        args_HNC['list_lambda'] = sorted([(-1)*lamb for lamb in args_HNC['list_lambda']])
    
    # reliable negatives
    
    for selector_ind, selector in enumerate(negative_selector_method):
        if selector=="skip":
            continue

        reliable_negatives, _ = negative_selector(y_pred, num_selected=num_reliable_negative, method=selector, random_seed=select_neg_seed)
        
        # construct new labeled-unlabeled sets and preidct
        X_train2 = np.concatenate((X_train, X_test[reliable_negatives,:])) 
        y_train2 = np.concatenate((y_train, np.array([0 for i in range(len(reliable_negatives))])))
        X_test2, y_test2 = np.delete(X_test, reliable_negatives, axis=0), np.delete(y_test, reliable_negatives)

        # reindex
        new_indices = np.concatenate((np.arange(labeled_size), labeled_size+reliable_negatives, labeled_size+np.delete(np.arange(X_test.shape[0]),reliable_negatives)))
        this_csr = neighbors_csr[new_indices, :][:, new_indices]

        hnc2 = HNC_pcut_2class_csr(**args_HNC)
        hnc2.fit(X_train2, y_train2)

        hnc2.fit_neighbor(this_csr)
        y_pred2 = hnc2.predict(X_test2) 

        y_actual_test = np.concatenate((y_test2, y_test[reliable_negatives]))
        y_actual_pred = np.concatenate((y_pred2, np.zeros((len(reliable_negatives), y_pred2.shape[1]))))

        predictions[:,y_pred.shape[1]:] = y_actual_pred

        if data_name != "other":

            for i in range(y_actual_pred.shape[1]):
                for j, metric in enumerate(metrics):
                    test_scores[selector_ind, j ,y_pred.shape[1]+i] = scorers[metric](y_actual_test, y_actual_pred[:,i])
        
        for i in range(y_actual_pred.shape[1]):
            positive_percentage[selector_ind,y_pred.shape[1]+i] = (y_actual_pred[:,i].sum()+labeled_size)/data_X.shape[0]

    # find the index of positive percentage closest to the true positive percentage
    closest_index = np.argmin(np.abs(positive_percentage - prior), axis=1)
    closest_percentage = positive_percentage[np.arange(num_selectors), closest_index]
    closest_prediction = predictions[:,closest_index]

    if data_name == "other":
        return {"pos": closest_percentage[0],  "predictions":closest_prediction}
    
    return {"pos": closest_percentage[0], "acc":test_scores[0, 0, closest_index[0]], "f1":test_scores[0, 1, closest_index[0]], "tpr":test_scores[0, 2, closest_index[0]], "tnr":test_scores[0, 3, closest_index[0]], "predictions":closest_prediction}

def main(arguments):

    args = process_args(arguments)

    if args.dataset == "other":
        if args.prior == None:
            raise ValueError("Please provide prior for this dataset")

    D = readData(args.dataset)

    if args.dataset == "other":
        X, y, numlabeled = data_prep_other(D) 
    else:
    
        if args.dataset == "news":
            X, y, numlabeled = news_data_prep(D, labeled_size=args.labeled, split_seed=args.seed)
        elif args.dataset == "mnist":
            X, y, numlabeled = mnist_data_prep(D, labeled_size=args.labeled, split_seed=args.seed)
        else:
            if args.labeledclass == 0:
                D['y'] = 1-D['y']
            if args.labeled < 1:
                numlabeled = int(np.sum(D['y'])*args.labeled)
            else:
                numlabeled = int(args.labeled)
            X, y, numlabeled = data_prep(D, labeled_size=numlabeled, split_seed=args.seed)
    
    # set up the parameters for HNC

    args_HNC = dict()
    args_HNC['list_lambda'] = [args.incr_lambda*i for i in range(args.numlambda)]
    args_HNC['k'] = 5
    args_HNC['epsilon'] = args.epslarge
    list_k = [5*i for i in range(1, (args.klarge//5)+1)]

    if D['X'].shape[0] >= 10000:
        args_HNC['list_lambda'] = [args.incr_lambda*i for i in range(args.numlambda)]
        args_HNC['epsilon'] = args.epssmall
        list_k = [5*i for i in range(1, (args.ksmall//5)+1)]
    
    print("True positive percentage: ", y.mean())

    if args.mode_numreliable == 0:
        numreliable = numlabeled
    elif args.mode_numreliable == 1:
        if args.dataset == "other":
            y_mean = args.prior
        else:
            y_mean = np.mean(y)
        numreliable = int((1-y_mean)*numlabeled/y_mean)
        print("numreliable: ", numreliable)
    
    list_pos_percent = np.zeros(len(list_k))
    list_acc = np.zeros(len(list_k))
    list_f1 = np.zeros(len(list_k))
    list_tpr = np.zeros(len(list_k))
    list_tnr = np.zeros(len(list_k))
    list_predictions = np.zeros((len(D['y'])-numlabeled, len(list_k)))

    prior_suffix = ""
    if args.priorshift != 1:
        if args.priorshift < 1:
            prior_suffix = "_0"+str(int(args.priorshift*10))
        else:
            prior_suffix = "_"+str(int(args.priorshift*10))

    fi = None

    if args.dataset == "other":
        # use the provided prior for datasets other than the ones used in the paper
        prior = args.prior 
    else:
        # automatically computed for datasets used in the paper
        prior = y.mean()*args.priorshift 

    start_time = time.perf_counter()
    for ind_k, k in enumerate(list_k):

        res = pu_hnc_csr(X, y, args.dataset, numlabeled, prior=prior, args_HNC=args_HNC, k_sparsify=k, feature_importance=args.featureweights, num_reliable_negative=numreliable, negative_selector_method=[args.select_reliable_method], select_neg_seed=args.negativeseed, fi=fi)
        print("Using k=", k)
        print("Positive percentage of chosen partition: ", res['pos'])
        list_pos_percent[ind_k] = res['pos']
        list_predictions[:, ind_k] = res['predictions'].reshape(-1)

        if args.dataset != "other":
            print("Accuracy: ", res['acc'])
            print("F1 score: ", res['f1'])
            
            list_acc[ind_k] = res['acc']
            list_f1[ind_k] = res['f1']
            list_tpr[ind_k] = res['tpr']
            list_tnr[ind_k] = res['tnr']
    
    ind_close = np.where(np.abs(list_pos_percent - prior) < 0.02)[0]
            
    if len(ind_close) == 0:
        ind_sel = np.argmin(np.abs(list_pos_percent - prior))
    else:
        ind_sel = ind_close[-1]
    tot_time = time.perf_counter() - start_time
    print(f"total runtime: {tot_time}")
    

    if args.featureweights:
        folder_name = f"results_hnc_{prior_suffix}/{args.dataset}_nr_{args.mode_numreliable}_{args.select_reliable_method}"
    else:
        folder_name = f"results_hnc_{prior_suffix}/{args.dataset}_nr_{args.mode_numreliable}_{args.select_reliable_method}_nofi"

    if not(os.path.exists(folder_name)):
        os.makedirs(folder_name)

    if args.dataset != "other":
        np.save(f"{folder_name}/acc_{args.seed}_{args.negativeseed}.npy", list_acc)
        np.save(f"{folder_name}/f1_{args.seed}_{args.negativeseed}.npy", list_f1)
        np.save(f"{folder_name}/tpr_{args.seed}_{args.negativeseed}.npy", list_tpr)
        np.save(f"{folder_name}/tnr_{args.seed}_{args.negativeseed}.npy", list_tnr)

    np.save("predictions.npy", list_predictions[:,ind_sel])

    if (args.dataset != "other") and (args.filename is not None):
        f = open(f"{folder_name}/{args.filename}", "a")
        f.write('acc {} f1 {} tpr {} tnr {} time {}\n'.format(list_acc[ind_sel], list_f1[ind_sel], list_tpr[ind_sel], list_tnr[ind_sel], tot_time)) 
        f.close()
    

if __name__ == '__main__':

    main(sys.argv[1:])
