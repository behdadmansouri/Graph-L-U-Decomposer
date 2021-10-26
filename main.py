import numpy as np

def soulless_get_input():
    n_and_m = input()
    n, m = n_and_m.split()
    n = int(n)
    m = int(m)
    matrix = np.zeros(shape=(n, n))
    bs = np.zeros(shape=(m, n))
    for i in range(int(n)):
        line = input()
        matrix[i] = line.split(' ')
    for j in range(int(m)):
        line = input()
        bs[j] = line.split(' ')
    return matrix, bs

def getLU(matrix):
    n = len(matrix[0])
    U = matrix.copy()
    L = np.eye(n, dtype=float)
    for i in range(n):
        factor = U[i + 1:, i] / U[i, i]
        L[i + 1:, i] = factor
        U[i + 1:] -= factor[:, np.newaxis] * U[i]
    return L, U

def forward_sub(L, b):
    # Ly=b
    n = L.shape[0]
    y = np.zeros_like(b, dtype=np.double)
    y[0] = b[0] / L[0, 0]
    for i in range(1, n):
        y[i] = (b[i] - np.dot(L[i, :i], y[:i])) / L[i, i]
    return y

def back_sub(U, y):
    #Ux=y
    n = U.shape[0]
    x = np.zeros_like(y, dtype=np.double)
    x[-1] = y[-1] / U[-1, -1]
    for i in range(n - 2, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i:], x[i:])) / U[i, i]
    return x


if __name__ == '__main__':
    matrix, bs = soulless_get_input()
    # matrix = np.array([[5, 6, 2], [4, 5, 2], [2, 4, 8]], dtype=float)
    # bs = np.array([[18, 7, 2], [4, 5, 8], [15, 7, 6], [11, 9, 5], [13, 12, 12]], dtype=float)
    L, U = getLU(matrix)

    for b in range(len(bs)):
        y = forward_sub(L, bs[b])
        x = back_sub(U, y)
        for i in x:
            print(round(i, 4), end="")
            print(" ", end="")

        print()
        # print(str(x).strip(' []'))
