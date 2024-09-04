import numpy as np
def gauss_elimination(A, b):
    n = len(b)
    Ab = np.hstack([A, b.reshape(-1, 1)])
    for i in range(n):
        max_row_index = np.argmax(np.abs(Ab[i:, i])) + i
        Ab[[i, max_row_index]] = Ab[[max_row_index, i]]

        for j in range(i + 1, n):
            factor = Ab[j, i] / Ab[i, i]
            Ab[j] -= factor * Ab[i]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (Ab[i, -1] - np.dot(Ab[i, i + 1:n], x[i + 1:n])) / Ab[i, i]
    return x
def find_free_and_pivot_variables(A):
    n, m = A.shape
    pivot_columns = []
    free_columns = list(range(m))
    for i in range(n):
        for j in range(m):
            if A[i, j] != 0 and j not in pivot_columns:
                pivot_columns.append(j)
                free_columns.remove(j)
                break
    return pivot_columns, free_columns
A = np.array([
    [2, 0, 0,],
    [0, 3, 0,],
    [0, 0, 1,],
], dtype=float) #just write here leftside cooficients
b = np.array([8, 9, -1], dtype=float) #write here rightside of equation cooficients
solution = gauss_elimination(A, b)
print("solution:", solution)
pivot_vars, free_vars = find_free_and_pivot_variables(A)
print("pivot:", pivot_vars)
print("free:", free_vars)