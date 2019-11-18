import numpy as np
import random


def generate_random_graph(N, alpha=0.5):  # random > alpha, then here is a edge.
    G = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            if random.random() < alpha:
                G[i][j] = 1
    return G


def getM(G, N):
    M = np.zeros((N, N))
    for i in range(N):
        D_i = sum(G[i])
        if D_i == 0:
            continue
        for j in range(N):
            M[j][i] = G[i][j] / D_i  # watch out! M_j_i instead of M_i_j
    return M


# Flow版本的PageRank
def pagerank2flow(M, N, T=300, tol=1e-6):
    R = np.ones(N) / N
    for _ in range(T):
        R1 = np.dot(M, R)
        if np.linalg.norm(R1 - R) < tol:
            break
        R = R1.copy()
    return R1


# Google版本的PageRank
def pagerank2google(M, N, T=300, tol=1e-6, beta=0.8):
    R = np.ones(N) / N
    teleport = np.ones(N) / N
    for _ in range(T):
        R1 = beta * np.dot(M, R) + (1-beta) * teleport
        if np.linalg.norm(R1 - R) < tol:
            break
        R = R1.copy()
    return R1


if __name__ == "__main__":
    G = generate_random_graph(10)
    M = getM(G, 10)
    values = pagerank2google(M, 10, T=2000, beta=0.8)
    print(values)
