"""Two-stage parametric graph-cut ranking on a fixed RNA-by-drug universe.

Uses the authors' simple HPF C solver through an array-based interface. Graph
construction and unaries follow the paper's objective, including both positive
affinity and the degree term in stage two. Returned ranks are not probabilities.
"""
import ctypes
from pathlib import Path
import threading

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parent
DEFAULT_LIBRARY = ROOT / 'research/inference_correction_20260916/hnc_author_bareHPF_v1.so'
_native_lock = threading.Lock()


def neighbor_graph(kernel, k=5):
    kernel = np.asarray(kernel, dtype=np.float64)
    if (kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1]
            or not np.isfinite(kernel).all() or not 0 < k < len(kernel)):
        raise ValueError('Invalid node kernel or neighbor count')
    matrix = kernel.copy()
    np.fill_diagonal(matrix, -np.inf)
    neighbors = np.argsort(-matrix, axis=1, kind='stable')[:, :k]
    rows = np.repeat(np.arange(len(matrix)), k)
    cols = neighbors.ravel()
    values = matrix[rows, cols]
    if np.any(values <= 0):
        raise ValueError('Selected graph similarities must be positive')
    graph = sparse.csr_matrix((values, (rows, cols)), shape=matrix.shape)
    graph = graph.maximum(graph.T)
    # Equal mean weighted degree for each biological axis; no label is read.
    graph /= float(graph.sum()) / len(matrix)
    return graph


def cartesian_graph(rna_kernel, drug_kernel, k=5):
    rna = neighbor_graph(rna_kernel, k)
    drug = neighbor_graph(drug_kernel, k)
    graph = sparse.kron(rna, sparse.eye(len(drug_kernel)), format='csr') + \
        sparse.kron(sparse.eye(len(rna_kernel)), drug, format='csr')
    graph.sort_indices()
    return graph


def solve_cuts(graph, positive, negative, parameters, library=DEFAULT_LIBRARY):
    """Return each node's first source-set cut index (one-based).

    For lambda <= 0, the unary is |lambda| * degree on the sink arc. For
    lambda >= 0, it is lambda * degree on the source arc. The C ABI implements
    constant + max(0, lambda * slope), which matters for negative parameters.
    """
    graph = sparse.csr_matrix(graph, dtype=np.float64)
    graph.sum_duplicates()
    graph.sort_indices()
    n = graph.shape[0]
    parameters = np.ascontiguousarray(parameters, dtype=np.float32)
    if (graph.shape != (n, n) or n >= 2**24 - 2 or not np.isfinite(graph.data).all()
            or np.any(graph.data < 0) or np.any(graph.diagonal())
            or (graph - graph.T).nnz and abs((graph - graph.T).data).max() > 1e-10):
        raise ValueError('Require a finite symmetric nonnegative graph without loops')
    if (parameters.ndim != 1 or len(parameters) < 2 or not np.isfinite(parameters).all()
            or np.any(np.diff(parameters) <= 0)
            or not (np.all(parameters >= 0) or np.all(parameters <= 0))):
        raise ValueError('Use an ordered parameter grid entirely on one side of zero')
    positive = np.unique(np.asarray(positive, dtype=np.int64))
    negative = np.unique(np.asarray(negative, dtype=np.int64))
    if (len(positive) < 1 or np.intersect1d(positive, negative).size
            or any(np.any(ids < 0) or np.any(ids >= n) for ids in (positive, negative))):
        raise ValueError('Invalid disjoint seed sets')
    free = np.ones(n, dtype=bool)
    free[positive] = False
    free[negative] = False
    unknown = np.flatnonzero(free)
    result = np.full(n, len(parameters)+1, dtype=np.int32)
    result[positive] = 1
    if not len(unknown):
        return result
    degree = np.asarray(graph.sum(axis=1)).ravel()[unknown]
    sw = np.asarray(graph[unknown][:, positive].sum(axis=1)).ravel()
    tw = np.asarray(graph[unknown][:, negative].sum(axis=1)).ravel()
    interior = graph[unknown][:, unknown].tocoo()
    m = len(unknown)
    arcs = np.zeros((interior.nnz + 2*m, 4), dtype=np.float32)
    arcs[:interior.nnz, 0] = interior.row
    arcs[:interior.nnz, 1] = interior.col
    arcs[:interior.nnz, 2] = interior.data
    source_arcs = arcs[interior.nnz:interior.nnz+m]
    sink_arcs = arcs[interior.nnz+m:]
    source_arcs[:, 0], source_arcs[:, 1], source_arcs[:, 2] = m, np.arange(m), sw
    sink_arcs[:, 0], sink_arcs[:, 1], sink_arcs[:, 2] = np.arange(m), m+1, tw
    if parameters[0] >= 0:
        source_arcs[:, 3] = degree
    else:
        sink_arcs[:, 3] = -degree
    lib = ctypes.CDLL(str(Path(library).resolve()))
    solve = lib.simparam_solve
    fp, ip = ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int)
    solve.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, fp,
                     ctypes.c_int, ctypes.c_int, fp, ctypes.POINTER(ip)]
    solve.restype = None
    lib.libfree.argtypes = [ctypes.c_void_p]
    lib.libfree.restype = None
    pointer = ip()
    # HPF uses global state, so concurrent calls within a process must serialize.
    with _native_lock:
        solve(m+2, len(arcs), len(parameters), arcs.ctypes.data_as(fp), m, m+1,
              parameters.ctypes.data_as(fp), ctypes.byref(pointer))
        if not pointer:
            raise RuntimeError('Native solver returned no result')
        try:
            local = np.ctypeslib.as_array(pointer, shape=(m+2,))[:m].copy()
        finally:
            lib.libfree(pointer)
    if np.any(local < 1) or np.any(local > len(parameters)+1):
        raise ValueError('Native cut index outside parameter grid')
    result[unknown] = local
    return result


def rank_from_breakpoints(breakpoints, grid_size):
    return (grid_size + 1 - np.asarray(breakpoints, dtype=np.float64)) / grid_size


def two_stage(graph, positive, negative_count, grid_size=501, library=DEFAULT_LIBRARY):
    """First-source-entry ranking, followed by likely-negative-constrained cuts.

    The likely-negative count is a registered design choice. It is not an
    estimate of latent biological prevalence, and selected negatives are not
    experimental negative labels. Ties use affinity to positive seeds, then
    the frozen RNA/drug dictionary order.
    """
    n = graph.shape[0]
    positive = np.unique(np.asarray(positive, dtype=np.int64))
    unknown = np.setdiff1d(np.arange(n), positive, assume_unique=True)
    if not 0 < negative_count < len(unknown):
        raise ValueError('Require a proper likely-negative subset')
    first = solve_cuts(graph, positive, [], np.linspace(-1., 0., grid_size), library)
    first_rank = rank_from_breakpoints(first, grid_size)
    affinity = np.asarray(graph[:, positive].sum(axis=1)).ravel()
    degree = np.asarray(graph.sum(axis=1)).ravel()
    affinity /= np.maximum(degree, 1e-15)
    order = np.lexsort((unknown, affinity[unknown], first_rank[unknown]))
    negative = unknown[order[:negative_count]]
    second = solve_cuts(graph, positive, negative, np.linspace(0., 1., grid_size), library)
    second_rank = rank_from_breakpoints(second, grid_size)
    return dict(first_breakpoints=first, second_breakpoints=second,
                first_rank=first_rank, full_rank=second_rank, negative=negative,
                positive=positive, grid_size=np.array(grid_size),
                negative_boundary_rank=np.array(first_rank[negative[-1]]),
                negative_boundary_ties=np.array(np.sum(first_rank[unknown] == first_rank[negative[-1]])))
