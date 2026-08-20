### Coefficients from Table 2
### Macromolecules 29, 23, 7602-7612, 1996

import numpy as np

# Instantiate static coefficient matrices ONCE at the module level
A1 = np.zeros(shape=(6, 3))
A1[2, 0] = -0.1222; A1[3, 0] = 0.3051; A1[4, 0] = -0.0711; A1[5, 0] = 0.0584
A1[2, 1] = 1.761;  A1[3, 1] = 2.252;  A1[4, 1] = -1.291; A1[5, 1] = 0.6994
A1[2, 2] = -26.04; A1[3, 2] = 20;     A1[4, 2] = 4.382;  A1[5, 2] = 1.594

A2 = np.zeros(shape=(6, 3))
A2[2, 1] = 0.1212; A2[3, 1] = -0.4169; A2[4, 1] = 0.1988; A2[5, 1] = 0.3435
A2[2, 2] = 0.0170; A2[3, 2] = -0.4731; A2[4, 2] = 0.1869; A2[5, 2] = 0.3350

B1 = np.zeros(shape=(3, 3))
B1[0, 0] = -0.0699; B1[1, 0] = -0.09;  B1[2, 0] = 0.2677
B1[0, 1] = 0.1342;  B1[1, 1] = 0.0138; B1[2, 1] = 0.1898
B1[0, 2] = -0.2020; B1[1, 2] = -0.0114; B1[2, 2] = 0.0123

B2 = np.zeros(shape=(3, 3))
B2[0, 1] = -0.5171; B2[1, 1] = -0.2028; B2[2, 1] = -0.3112
B2[0, 2] = 0.6950;  B2[1, 2] = -0.3238; B2[2, 2] = -0.5403

def get_Ai(i, L, b):
    # Vectorized computation supporting 1D and 2D arrays
    L_b = L / b
    exp_10 = np.exp(-10 * b / L)
    exp_2 = np.exp(-2 * L_b)
    
    return (
        (A1[i, 0] * exp_10 + A2[i, 0] * exp_2) + 
        (A1[i, 1] * (L_b**-1) * exp_10 + A2[i, 1] * L_b * exp_2) + 
        (A1[i, 2] * (L_b**-2) * exp_10 + A2[i, 2] * (L_b**2) * exp_2)
    )

def get_Bi(i, L, b):
    L_b = L / b
    exp_2 = np.exp(-2 * L_b)
    
    return (
        (B1[i, 0] + B2[i, 0] * exp_2) + 
        (B1[i, 1] * (L_b**-1) + B2[i, 1] * L_b * exp_2) + 
        (B1[i, 2] * (L_b**-2) + B2[i, 2] * (L_b**2) * exp_2)
    )
