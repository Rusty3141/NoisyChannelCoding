import numpy as np
import networkx as nx
from networkx.algorithms import bipartite
import matplotlib.pyplot as plt
import scipy as sp


def ldpc_sim(N, columnWeight, rowWeight):
    R = int(N * columnWeight / rowWeight)

    HFirstSubmatrix = np.zeros((int(R/columnWeight), N))

    for i in range(int(R/columnWeight)):
        for j in range(rowWeight):
            HFirstSubmatrix[i, rowWeight*i+j] = 1

    secondSubmatrixPermutation = np.random.permutation(
        HFirstSubmatrix.shape[1])
    thirdSubmatrixPermutation = np.random.permutation(
        HFirstSubmatrix.shape[1])

    H = np.vstack((HFirstSubmatrix,
                   HFirstSubmatrix[:, secondSubmatrixPermutation], HFirstSubmatrix[:, thirdSubmatrixPermutation]))

    finalRowPermutation = np.random.permutation(H.shape[0])
    finalColumnPermutation = np.random.permutation(H.shape[1])

    H = H[:, finalColumnPermutation]
    H = H[finalRowPermutation, :]

    H = np.array([[0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1],
                  [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0],
                  [0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0],
                  [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1],
                  [0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1],
                  [1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0],
                  [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                  [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0]])
    print(H)
    G = nx.Graph()

    G.add_nodes_from([f"C{i}" for i in range(R)], bipartite=0)
    G.add_nodes_from(range(N), bipartite=1)

    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            if H[i, j] == 1:
                G.add_edge(f"C{i}", j)

    pos = nx.drawing.layout.bipartite_layout(
        G, [f"C{i}" for i in range(R)], align='horizontal')
    nx.draw_networkx_edges(G, pos,
                           width=0.4)

    for i, node in enumerate(G.nodes()):
        G.nodes[node]['shape'] = 'o' if i < R else 's'

    for shape in ['o', 's']:
        node_list = [node for node in G.nodes() if G.nodes[node]
                     ['shape'] == shape]

        nx.draw_networkx_nodes(G, pos,
                               nodelist=node_list,
                               node_size=75 if shape == 's' else 100,
                               node_color='black',
                               node_shape=shape)

    ax = plt.gca()

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    plt.xlabel("Check (Constraint) Nodes")
    plt.title("Message (Data) Nodes")

    plt.savefig("TannerGraph.png", dpi=300, bbox_inches='tight', pad_inches=0)

    Hcopy = []

    while (True):
        Hcopy = H[:, :]
        rowsToDrop = np.random.choice(
            H.shape[0], R-np.linalg.matrix_rank(H), replace=False)
        Hcopy = np.delete(Hcopy, rowsToDrop, 0)
        if (np.linalg.matrix_rank(Hcopy) == np.linalg.matrix_rank(H)):
            break

    K = N-Hcopy.shape[0]
    print(f"N={N}, R={R}, K={K}")

    aux = []
    while (True):
        aux = Hcopy[:, np.random.permutation(Hcopy.shape[1])]
        if (np.linalg.det(aux[:, :N-K]) % 2 == 1):
            break

    Hcopy = aux[:]
    A = Hcopy[:, :N-K]
    B = Hcopy[:, N-K:]

    s = np.array([1, 0, 0, 1, 1, 0])

    P, L, U = sp.linalg.lu(A)
    P = P % 2
    L = L % 2
    U = U % 2
    print(P, L, U)
    xBar = P@B@s % 2
    y = np.zeros(L.shape[1])

    for i in range(len(y)):
        y[i] = xBar[i]-sum([L[i, j]*y[j] for j in range(i)])

    c = np.zeros(U.shape[1])
    for i in range(len(c)):
        c[len(c)-1-i] = y[len(c)-1-i] - \
            sum([U[len(c)-1-i, len(c)-1-j]*c[len(c)-1-j] for j in range(i)])

    cs = np.concatenate((c, s)) % 2
    invC = np.linalg.inv(A)@B@s
    print(H@np.concatenate((invC, s)))
    print(H@cs)
    print(np.linalg.inv(A)@B@s, c)


def reduce(X):
    A = np.copy(X)
    m, n = A.shape

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

            for i in range(m):
                if i != pivot_old and A[i, j]:
                    A[i, :] = abs(A[i, :]-A[pivot_old, :])

        if pivot_old == m-1:
            break

    return A


def main():
    ldpc_sim(16, 3, 4)


if __name__ == "__main__":
    main()
