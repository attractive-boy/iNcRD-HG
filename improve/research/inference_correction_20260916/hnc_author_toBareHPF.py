"""
this code is from https://riot.ieor.berkeley.edu/Applications/Pseudoflow/parametric.html

    toBareHPF allows a Python program to call the simple parametric cut C program
    directly by passing the following arguments:
    1. Arcs (edges) as a Python dictionary eds[(i, j)] = (a_ij, b_ij) where i and j are the
       endpoint nodes of the arc and the capacity is max(0,a_ij + lambda * b_ij).
       In the case of sink adjacent edges, b_i must be negative
       Note that nodes are zero indexed (i.e. they go from 0 to n-1)
       Also note that edges are considered in only one direction, so
       if necessary, edges (i,j) and (j,i) must be added
    2. Number of nodes on the network, including the source and sink
    3. ID of the source node
    4. ID of the sink node
    5. Parameter (lambda) values as a Python list of p values [l_0, l_1, ..., l_(p-1)]
       Remarks: The parameter values need to be in the order of the cuts.
       So usually by decreasing values of lambda.

    toBareHPF returns a Python list of integers.  The size of the list equals n,
    the number of nodes in the graph.  The value at position k, k=0,...,n-1, is
    the breakpoint (cut index) at which node k entered the source set.

To compile the C program BareHPF.c to use with C-types:
gcc -fPIC -shared -o bareHPF.so bareHPF.c -- Linux
gcc -shared -Wl,-install_name bareHPF.so -o bareHPF.so -fPIC bareHPF.c --MACOS
Remark: Not sure what is the correct command for Windows.
"""


from ctypes import c_int, cast, byref, POINTER, cdll, c_float
import os
import time

# path = os.getcwd()
# simHPF = cdll.LoadLibrary(os.path.join(path, 'bareHPF.so'))
# print("simHPF ", simHPF)


def _c_arr(c_type, size, init):
    x = c_type * size
    return x(*init)


def _create_c_input(flatEds, n_nodes, sourceID, sinkID, lambdas):
    # print('typApp ', typApp)
    nNodes = n_nodes 
    nArcs = int(len(flatEds)/4)
    nParams = len(lambdas)
    c_numNodes = c_int(nNodes)
    c_numArcs = c_int(nArcs)
    c_numParams = c_int(nParams)
    c_source = c_int(sourceID)
    c_sink = c_int(sinkID)
    c_eds = _c_arr(c_float, nArcs*4, flatEds)
    c_params = _c_arr(c_float, nParams, lambdas)
    c_breakpoints = POINTER(c_int)()

    return {
        "numNodes": c_numNodes,
        "numArcs": c_numArcs,
        "numLambda": c_numParams,
        "eds": c_eds,
        "source": c_source,
        "sink": c_sink,
        "params": c_params,
        "bps": c_breakpoints,
    }


def _solve(c_input, simHPF):
    # global simHPF
    simHPFSolve = simHPF.simparam_solve
    simHPFSolve.argtypes = [
        c_int,
        c_int,
        c_int,
        POINTER(c_float),
        c_int,
        c_int,
        POINTER(c_float),
        POINTER(POINTER(c_int)),
        ]

    simHPFSolve(
        c_input["numNodes"],
        c_input["numArcs"],
        c_input["numLambda"],
        cast(byref(c_input["eds"]), POINTER(c_float)),
        c_input["source"],
        c_input["sink"],
        cast(byref(c_input["params"]), POINTER(c_float)),
        byref(c_input["bps"]),
    )


def flattenNds(dic):
    flatDic = []
    for key in dic:
        fl = [key]
        for i in dic[key]:
            fl.append(i)
        if len(fl) > 2:
            tmp = fl[2]
            fl[2] = fl[1]
            fl[1] = tmp
        flatDic += fl
    # print(flatDic[:20])
    return flatDic


def flattenEds(dic):
    #flatDic = [ i, j, dic[(i,j)][0], dic[(i,j)][1] for (i,j) in dic  ]

    flatDic = []
    for (i, j) in dic:
        fl = [i, j, dic[(i, j)][0] , dic[(i, j)][1] ]
        flatDic += fl
    return flatDic


def _read_output(c_input, numNodes):
    breakpoints = [c_input["bps"][i] for i in range(numNodes)]
    return breakpoints


def _cleanup(c_input, simHPF):
    # global simHPF
    cleanbp = simHPF.libfree
    cleanbp(c_input["bps"])


def toBareHPF(eds, n_nodes, sourceID, sinkID, lambdas):
    '''
     - eds is the dictionary of edges or arcs in form eds[(i,j)] = (a_ij, b_ij) 
        nodes are 0 indexed, i.e their IDs go from 0 to n_nodes-1
        consider that the arc will have capacity a_ij + lambda * b_ij
     - n_nodes is the number of nodes in the network, including source and sink
     - sourceID and sinkID correspond to the IDs of the
        source and sink nodes
     - lambdas is the list of parameter (lambda) values
        value in lambda list must be such that capacities in source
        adjacent edges will be monotone non decreasing
    '''
    # print('I enter toBareHPF')
    path = os.getcwd()
    simHPF = cdll.LoadLibrary(os.path.join(path, 'bareHPF.so'))
    # print('cdll library loaded')
    stTime = time.time()
    flatEds = flattenEds(eds)
    c_input = _create_c_input(
        flatEds, n_nodes, sourceID, sinkID, lambdas)
    #print('Time to translate data from Python to C took %.4f seconds'
    #      % (time.time()-stTime))
    _solve(c_input, simHPF)
    # print('Executed the solve')
    breakpoints = _read_output(c_input, n_nodes)
    # print('Got the breakpoints')
    _cleanup(c_input, simHPF)
    # print('Finished cleanup of the breakpoints')
    del simHPF
    return breakpoints
