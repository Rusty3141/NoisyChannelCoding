import numpy as np
from scipy.sparse import csr_matrix

import math
import numbers
import numpy as np
import scipy
from scipy.stats import norm
pi = math.pi


def int2bitarray(n, k):
    """Change an array's base from int (base 10) to binary (base 2)."""
    binary_string = bin(n)
    length = len(binary_string)
    bitarray = np.zeros(k, 'int')
    for i in range(length - 2):
        bitarray[k - i - 1] = int(binary_string[length - i - 1])

    return bitarray


def bitarray2int(bitarray):
    """Change array's base from binary (base 2) to int (base 10)."""
    bitstring = "".join([str(i) for i in bitarray])

    return int(bitstring, 2)


def binaryproduct(X, Y):
    """Compute a matrix-matrix / vector product in Z/2Z."""
    A = X.dot(Y)
    try:
        A = A.toarray()
    except AttributeError:
        pass
    return A % 2


def gaussjordan(X, change=0):
    """Compute the binary row reduced echelon form of X.
    Parameters
    ----------
    X: array (m, n)
    change : boolean (default, False). If True returns the inverse transform
    Returns
    -------
    if `change` == 'True':
        A: array (m, n). row reduced form of X.
        P: tranformations applied to the identity
    else:
        A: array (m, n). row reduced form of X.
    """
    A = np.copy(X)
    m, n = A.shape

    if change:
        P = np.identity(m).astype(int)

    pivot_old = -1
    for j in range(n):
        filtre_down = A[pivot_old+1:m, j]
        pivot = np.argmax(filtre_down)+pivot_old+1

        if A[pivot, j]:
            pivot_old += 1
            if pivot_old != pivot:
                aux = np.copy(A[pivot, :])
                A[pivot, :] = A[pivot_old, :]
                A[pivot_old, :] = aux
                if change:
                    aux = np.copy(P[pivot, :])
                    P[pivot, :] = P[pivot_old, :]
                    P[pivot_old, :] = aux

            for i in range(m):
                if i != pivot_old and A[i, j]:
                    if change:
                        P[i, :] = abs(P[i, :]-P[pivot_old, :])
                    A[i, :] = abs(A[i, :]-A[pivot_old, :])

        if pivot_old == m-1:
            break

    if change:
        return A, P
    return A


def binaryrank(X):
    """Compute rank of a binary Matrix using Gauss-Jordan algorithm."""
    A = np.copy(X)
    m, n = A.shape

    A = gaussjordan(A)

    return sum([a.any() for a in A])


def f1(y, sigma):
    """Compute normal density N(1,sigma)."""
    f = norm.pdf(y, loc=1, scale=sigma)
    return f


def fm1(y, sigma):
    """Compute normal density N(-1,sigma)."""

    f = norm.pdf(y, loc=-1, scale=sigma)
    return f


def _bitsandnodes(H):
    """Return bits and nodes of a parity-check matrix H."""
    if type(H) != scipy.sparse.csr_matrix:
        bits_indices, bits = np.where(H)
        nodes_indices, nodes = np.where(H.T)
    else:
        bits_indices, bits = scipy.sparse.find(H)[:2]
        nodes_indices, nodes = scipy.sparse.find(H.T)[:2]
    bits_histogram = np.bincount(bits_indices)
    nodes_histogram = np.bincount(nodes_indices)

    return bits_histogram, bits, nodes_histogram, nodes


def bits2i(H, i):
    """Compute list of variables (bits) connected to Parity node i."""
    if type(H) != scipy.sparse.csr_matrix:
        m, n = H.shape
        return list(np.where(H[i])[0])

    indj = H.indptr
    indi = H.indices

    return [indi[a] for a in range(indj[i], indj[i+1])]


def nodes2j(H, j):
    """Compute list of nodes (PC equations) connecting variable j."""
    return bits2i(H.T, j)


def bitsandnodes(H):
    """Return bits and nodes of a parity-check matrix H."""
    m, n = H.shape

    bits = [bits2i(H, i) for i in range(m)]
    nodes = [nodes2j(H, j) for j in range(n)]
    bits = np.array(bits)
    nodes = np.array(nodes)
    return bits, nodes


def incode(H, x):
    """Compute Binary Product of H and x."""
    return (binaryproduct(H, x) == 0).all()


def gausselimination(A, b):
    """Solve linear system in Z/2Z via Gauss Gauss elimination."""
    if type(A) == scipy.sparse.csr_matrix:
        A = A.toarray().copy()
    else:
        A = A.copy()
    b = b.copy()
    n, k = A.shape

    for j in range(min(k, n)):
        listedepivots = [i for i in range(j, n) if A[i, j]]
        if len(listedepivots):
            pivot = np.min(listedepivots)
        else:
            continue
        if pivot != j:
            aux = (A[j, :]).copy()
            A[j, :] = A[pivot, :]
            A[pivot, :] = aux

            aux = b[j].copy()
            b[j] = b[pivot]
            b[pivot] = aux

        for i in range(j+1, n):
            if A[i, j]:
                A[i, :] = abs(A[i, :]-A[j, :])
                b[i] = abs(b[i]-b[j])

    return A, b


def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance
    Parameters
    ----------
    seed : None | int | instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, numbers.Integral):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)


def parity_check_matrix(n_code, d_v, d_c, seed=None):
    """
    Build a regular Parity-Check Matrix H following Callager's algorithm.
    Parameters
    ----------
    n_code: int, Length of the codewords.
    d_v: int, Number of parity-check equations including a certain bit.
        Must be greater or equal to 2.
    d_c: int, Number of bits in the same parity-check equation. d_c Must be
        greater or equal to d_v and must divide n.
    seed: int, seed of the random generator.
    Returns
    -------
    H: array (n_equations, n_code). LDPC regular matrix H.
        Where n_equations = d_v * n / d_c, the total number of parity-check
        equations.
    """
    rng = check_random_state(seed)

    if d_v <= 1:
        raise ValueError("""d_v must be at least 2.""")

    if d_c <= d_v:
        raise ValueError("""d_c must be greater than d_v.""")

    if n_code % d_c:
        raise ValueError("""d_c must divide n for a regular LDPC matrix H.""")

    n_equations = (n_code * d_v) // d_c

    block = np.zeros((n_equations // d_v, n_code), dtype=int)
    H = np.empty((n_equations, n_code))
    block_size = n_equations // d_v

    # Filling the first block with consecutive ones in each row of the block

    for i in range(block_size):
        for j in range(i * d_c, (i+1) * d_c):
            block[i, j] = 1
    H[:block_size] = block

    # reate remaining blocks by permutations of the first block's columns:
    for i in range(1, d_v):
        H[i * block_size: (i + 1) * block_size] = rng.permutation(block.T).T
    H = H.astype(int)
    return H


def coding_matrix(H, sparse=True):
    """Return the generating coding matrix G given the LDPC matrix H.
    Parameters
    ----------
    H: array (n_equations, n_code). Parity check matrix of an LDPC code with
        code length `n_code` and `n_equations` number of equations.
    sparse: (boolean, default True): if `True`, scipy.sparse format is used
        to speed up computation.
    Returns
    -------
    G.T: array (n_bits, n_code). Transposed coding matrix.
    """
    if type(H) == csr_matrix:
        H = H.toarray()
    n_equations, n_code = H.shape

    # DOUBLE GAUSS-JORDAN:

    Href_colonnes, tQ = gaussjordan(H.T, 1)

    Href_diag = gaussjordan(np.transpose(Href_colonnes))

    Q = tQ.T

    print(Href_colonnes, tQ)

    n_bits = n_code - Href_diag.sum()

    Y = np.zeros(shape=(n_code, n_bits)).astype(int)
    Y[n_code - n_bits:, :] = np.identity(n_bits)

    print(Q, Y)

    if sparse:
        Q = csr_matrix(Q)
        Y = csr_matrix(Y)

    tG = binaryproduct(Q, Y)

    return tG


H = parity_check_matrix(16, 3, 4)
print(H)
G = coding_matrix(H)
print(gaussjordan(H))
print(G)
