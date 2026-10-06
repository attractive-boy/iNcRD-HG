# Positive-unlabeled Learning using Pairwise Similarities and Parametric Minimum Cuts

*Nitayanont, T., Hochbaum, D.S.: Positive-unlabeled learning using pairwise similarity and parametric minimum cuts. In: Proceedings of the 16th International Joint Conference on Knowledge Discovery, Knowledge Engineering and Knowledge Management, vol. 1, pp. 60–71 (2024)* </br>
[Published in KDIR 2024](https://www.scitepress.org/Papers/2024/129481/129481.pdf)
***

<!--
## Abstract

Positive-unlabeled (PU) learning is a binary classification problem where the labeled set contains only positive class samples. 

Our PU learning method, called **2-HNC**, is based on a network flow-based method called Hochbaum’s Normalized Cut (HNC) that partitions samples, both labeled and unlabeled, into two sets to achieve high intra-similarity and low inter-similarity, with a tradeoff parameter to balance these two goals. 
HNC is solved, for all tradeoff values, as a parametric minimum cut problem on an associated graph producing multiple optimal partitions, which are nested for increasing tradeoff values.

*2-HNC* runs in two stages.
Stage 1 identifies optimal data partitions for all tradeoff values, using only positive labeled samples.
Stage 2 first ranks unlabeled samples by their likelihood of being negative, according to the sequential order of partitions from stage 1, and then uses the likely-negative along with positive samples to run HNC.
Among all generated partitions in both stages, the partition whose positive fraction is closest to the true positive fraction $\pi$, which is often given as a prior to PU learning methods, is selected. An experimental study demonstrates that *2-HNC* is highly competitive compared to state-of-the-art methods.

*** 
--> 
## Our implementation of 2-HNC

Given the set of positive labeled samples and the set of unlabeled samples, which is a mixture of samples from the positive and the negative classes, the goal of 2-HNC is to predict the labels of the unlabeled samples.

The code takes the following as inputs
* the feature array of positive labeled samples
* the feature array of unlabeled samples, of which we want to predict the labels
* the prior $\pi$ of the fraction of positive samples in the data, as estimated by the user

The feature arrays of positive labeled samples and unlabeled samples need to be prepared in the directory as numpy arrays, and need to be named *X_labeled.npy* and *X_unlabeled.npy*, respectively. The two arrays should be formatted such that the element on row *i* and column *j* is the value of the *j*-th feature of sample *i*.

The prior $\pi$, or its estimate if unknown, is to be provided to the 2-HNC method through the command line. This is explained in the following section on how to run the code of 2-HNC.

Our code reads in the data from the two numpy arrays. Then, it runs the first stage of 2-HNC, ranks unlabeled samples according to the their likelihood of being negative and runs the second stage of 2-HNC, before selecting one data partition (from all that are generated in both stages) as prediction, based on $\pi$. 

The output is 
* the label predictions of unlabeled samples, saved as a numpy array *predictions.npy*. The $i$-th entry of the array indicates the label prediction for the $i$-th unlabeled sample, where '1' indicates the positive prediction and '0' the negative prediction.

This implementation of 2-HNC involves the use of the simple HPF parametric minimum cut solver, which can be found at [https://riot.ieor.berkeley.edu/Applications/Pseudoflow/parametric.html](https://riot.ieor.berkeley.edu/Applications/Pseudoflow/parametric.html), under the section of *HPF Simple Parametric Maximum Flow Solver Version 2.0 (Linear parametric functions, and input of parameter values)*.

***
## To run the code of 2-HNC

First, compile the c code of the simple HPF parametric minimum cut solver.
```{python}
gcc -fPIC -shared -o bareHPF.so bareHPF.c
```

To run the 2-HNC method, create/store the data files first, as explained above, then run the main python file, *HNCPU.py*, and specify the prior $\pi$ in the command line. <br>
For example, to set the prior $\pi$ to $0.6$, run the following
```{python}
python HNCPU.py -pr 0.6
```

The command line in the instruction here runs 2-HNC with the default values of parameters that are provided in the code. 
Next, in the section below, we show how values of these parameters can be modified by users.

## Modifying values of parameters in the 2-HNC method to be different from their default values

The 2-HNC method solves the PU learning problem as a network flow problem on the graph representation of the data. In the graph, nodes that represent data samples are connected via edges that carry *pairwise similarity weights*. There are parameters in the construction of this graph that users can set values to be different from their default values that we use in the paper. These parameters include

1. $\sigma$ in the pairwise similarity weight
    - The pairwise similarity weight between sample $i$ and $j$ that we use in this work is $\exp{(-\frac{d_{ij}^2}{2\sigma^2})}$, where $d_{ij}$ is the distance between samples $i$ and $j$. In this implementation, the default value of $\sigma$ for a dataset of size smaller than $10000$ is $0.75$ whereas the default value for a dataset of size larger than or equal to $10000$ is $0.25$. 
    - To change the value of $\sigma$ for a dataset of size smaller than $10000$, for example, to $0.5$, run the following line
      ```{python}
      python HNCPU.py -pr 0.6 -el 0.5
      ```
    - To change the value of $\sigma$ for a dataset of size larger than or equal to $10000$, for example, to $0.15$, run the following line
      ```{python}
      python HNCPU.py -pr 0.6 -es 0.15
      ```

2. Feature importance weight in the computation of the pairwise distance $d_{ij}$
    - The pairwise distance is computed by default as the Euclidean distance weighted by the feature importance weight. The feature importance weight is obtained from the tree-based PU ET model [https://github.com/jonathanwilton/PUExtraTrees](https://github.com/jonathanwilton/PUExtraTrees). Our implementation also allows users to use the standard, unweighted Euclidean distance.
    - To use the unweighted Euclidean distance, run the following line
      ```{python}
      python HNCPU.py -pr 0.6 --no-fi
      ```
      
3. *k-nearest neighbor (kNN)* sparsification
    - kNN sparsification is applied to prevent highly unbalanced partition of the graph. See the paper for the detailed explanation.
    - The default value of $k$ in the code for a dataset of size smaller than $10000$ is $15$. The default value for a dataset of size larger than or equal to $10000$ is $5$.
    - To change the value of $k$ for a dataset of size smaller than $10000$, for example, to $20$, run the following line
      ```{python}
      python HNCPU.py -pr 0.6 -kl 20
      ```
   - To change the value of $k$ for a dataset of size larger than or equal to $10000$, for example, to $10$, run the following line
      ```{python}
      python HNCPU.py -pr 0.6 -ks 10
      ```

In addition to the graph construction, users can also choose the set of values of the tradeoff parameter ($\lambda$) in the objective function of HNC to be different from the default values in our implementation of 2-HNC. 

4. Tradeoff parameter ($\lambda$) values 
    - The default list of values of $\lambda$ in the implementation is a list of $501$ evenly spaced increments of $0.001$, starting from $0$ and ending at $0.500$.
    - Users can change the number and the increment of values of $\lambda$ in the list. For instance, to use a list of $26$ evenly spaced increments of $0.004$, starting from $0$ and ending at $0.100$, that is $[0, 0.004, 0.008, ..., 0.100]$, run the following line
      ```{python}
      python HNCPU.py -pr 0.6 -nl 26 -il 0.004
      ```
