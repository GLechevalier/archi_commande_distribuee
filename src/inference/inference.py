import matplotlib.pyplot as plt
import numpy as np
from ..lib.potential import Potential
from scipy.optimize import curve_fit


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


def diff(L, Ly_noise):
    Ly_true = evaluate_n_points_from_list(L)
    sum = 0
    for i in range(len(L)):
        sum += (Ly_noise[i] - Ly_true[i]) ** 2
    return sum


L = L.transpose()
data = Ly_true[:].reshape(1, len(Ly_true))
print(L)
print(data)

# Initial guess for parameters
initial_guess = (1, 0, 0, 1, 1, 0, 0)


# Define the 2D Gaussian function
def twoD_Gaussian(coordinates, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    x, y = coordinates
    xo = float(xo)
    yo = float(yo)
    # Calculate the rotation components
    a = (np.cos(theta) ** 2) / (2 * sigma_x**2) + (np.sin(theta) ** 2) / (
        2 * sigma_y**2
    )
    b = -np.sin(2 * theta) / (4 * sigma_x**2) + np.sin(2 * theta) / (4 * sigma_y**2)
    c = (np.sin(theta) ** 2) / (2 * sigma_x**2) + (np.cos(theta) ** 2) / (
        2 * sigma_y**2
    )
    # Compute the Gaussian function
    g = offset + np.log10(
        amplitude
        * np.exp(
            -(a * ((x - xo) ** 2) + 2 * b * (x - xo) * (y - yo) + c * ((y - yo) ** 2))
        )
    )
    return g.ravel()  # Flatten the 2D array to 1D for curve_fit


# Perform the curve fitting
popt, pcov = curve_fit(twoD_Gaussian, L, data.ravel(), p0=initial_guess)

popt_dict = {
    "amplitude": popt[0],
    "xo": popt[1],
    "yo": popt[2],
    "sigma_x": popt[3],
    "sigma_y": popt[4],
    "theta": popt[5],
    "offset": popt[6],
}

print("Fitted parameters:", popt_dict)
