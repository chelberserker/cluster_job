### Coefficients from Table 2
### Macromolecules 29, 23, 7602-7612, 1996

import numpy as np

def get_a1():
    A1 = np.zeros(shape=(6, 3))
    A1[2, 0] = -0.1222
    A1[3, 0] = 0.3051
    A1[4, 0] = -0.0711
    A1[5, 0] = 0.0584
    A1[2, 1] = 1.761
    A1[3, 1] = 2.252
    A1[4, 1] = -1.291
    A1[5, 1] = 0.6994
    A1[2, 2] = -26.04
    A1[3, 2] = 20
    A1[4, 2] = 4.382
    A1[5, 2] = 1.594

    return A1

def get_a2():
    A2 = np.zeros(shape=(6, 3))
    A2[2, 1] = 0.1212
    A2[3, 1] = -0.4169
    A2[4, 1] = 0.1988
    A2[5, 1] = 0.3435
    A2[2, 2] = 0.0170
    A2[3, 2] = -0.4731
    A2[4, 2] = 0.1869
    A2[5, 2] = 0.3350

    return A2

def get_b1():
    B1 = np.zeros(shape=(3, 3))
    B1[0, 0] = -0.0699
    B1[1, 0] = -0.09
    B1[2, 0] = 0.2677
    B1[0, 1] = 0.1342
    B1[1, 1] = 0.0138
    B1[2, 1] = 0.1898
    B1[0, 2] = -0.2020
    B1[1, 2] = -0.0114
    B1[2, 2] = 0.0123

    return B1

def get_b2():
    B2 = np.zeros(shape=(3, 3))
    B2[0, 1] = -0.5171
    B2[1, 1] = -0.2028
    B2[2, 1] = -0.3112
    B2[0, 2] = 0.6950
    B2[1, 2] = -0.3238
    B2[2, 2] = -0.5403

    return B2

def get_Ai(i, L, b):
    a1 = get_a1()
    a2 = get_a2()

    Ai = 0
    for j in range(a1.shape[1]):
        Ai += a1[i,j]*(L/b)**(-j)*np.exp(-10*b/L) + a2[i,j]*(L/b)**(j)*np.exp(-2*L/b)

    return Ai

def get_Bi(i, L, b):
    b1 = get_b1()
    b2 = get_b2()

    Bi = 0
    for j in range(b1.shape[1]):
        Bi += b1[i,j]*(L/b)**(-j) + b2[i,j]*(L/b)**(j)*np.exp(-2*L/b)

    return Bi


