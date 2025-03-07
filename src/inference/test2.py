import numpy as np
from scipy.optimize import curve_fit


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
    g = offset + amplitude * np.exp(
        -(a * ((x - xo) ** 2) + 2 * b * (x - xo) * (y - yo) + c * ((y - yo) ** 2))
    )
    return g.ravel()  # Flatten the 2D array to 1D for curve_fit


# Generate synthetic data

n_res = 200
xmin = -10
xmax = 10
ymin = -10
ymax = 10

x = np.linspace(xmin, xmax, n_res)
y = np.linspace(ymin, ymax, n_res)
x, y = np.meshgrid(x, y)

# True parameters for synthetic data
true_params = {
    "amplitude": 10,
    "xo": 5,
    "yo": -5,
    "sigma_x": 10,
    "sigma_y": 2,
    "theta": 1.7,
    "offset": 20,
}


def sample(x, y, percentage):
    largeur = len(x[0])

    L = []

    for i in range(len(x)):
        for j in range(largeur):
            if np.random.rand() > 1 - percentage:

                L.append([x[i, j], y[i, j]])

    np.random.shuffle(L)
    L = np.transpose(np.array(L))
    return L


L = sample(x, y, 0.1)

n_data = len(L[0])

data = twoD_Gaussian(L, **true_params).reshape(1, n_data)

print(data)

# Add noise to simulate observations
noise = 0.00 * np.random.normal(size=data.shape)
data_noisy = data + noise

# Initial guess for parameters
initial_guess = (1, 0, 0, 1, 1, 0, 0)

# Perform the curve fitting
popt, pcov = curve_fit(twoD_Gaussian, L, data_noisy.ravel(), p0=initial_guess)
print("Fitted parameters:", popt)
