import matplotlib.pyplot as plt
import numpy as np
from ..lib.potential import Potential
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel


pot1 = Potential(difficulty=1, random=True)

fig2, ax2 = pot1.plot(2)

epsilon = 1.0


def get_true_parameters(pot: Potential):
    moy_1 = pot.mu1
    x_moy = moy_1[0]
    y_moy = moy_1[1]
    sigma_1 = pot.gaussian1.cov[0, 0]
    sigma_2 = pot.gaussian1.cov[1, 1]
    return x_moy, y_moy, sigma_1, sigma_2


x_moy, y_moy, sigma_1, sigma_2 = get_true_parameters(pot1)
print(x_moy, y_moy)
print(sigma_1)
print(sigma_2)


def generate_n_points(n: int):
    L = []
    for i in range(n):
        L.append(
            [
                np.random.uniform(pot1.xmin, pot1.xmax),
                np.random.uniform(pot1.ymin, pot1.ymax),
            ]
        )
    return L


def evaluate_n_points(n: int):
    return_list = []
    L = generate_n_points(n=n)
    for i in range(n):
        return_list.append(pot1.value(L[i]))
    return L, return_list


def evaluate_n_points_from_list(L):
    return_list = []
    for i in range(len(L)):
        return_list.append(pot1.value(L[i]))
    return return_list


def generate_n_points_noise(n: int):
    L, Ly = evaluate_n_points(10)
    for i in range(len(Ly)):
        Ly[i] += np.random.normal()
    return L, Ly


L, Ly_noise = generate_n_points_noise(100)
L = np.array(L)
Ly_true = evaluate_n_points_from_list(L)
Ly_true = np.array(Ly_true)
print(L), print(Ly_true)


def diff(L, Ly_noise):
    Ly_true = evaluate_n_points_from_list(L)
    sum = 0
    for i in range(len(L)):
        sum += (Ly_noise[i] - Ly_true[i]) ** 2
    return sum
