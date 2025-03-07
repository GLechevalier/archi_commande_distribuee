import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

# Assuming you have your training data (X_train, y_train)
# and your test data X_test:
X_train = np.array([[-1, 0], [0, 0], [1, 0], [0, 1], [0, -1]])
y_train = np.array([0, 1, 0, 0, 0])
X_test = np.array([[1, 0], [2, 1]])

# Define the kernel: RBF plus noise
kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)

# Create and fit the Gaussian Process model
gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=0)
gpr.fit(X_train, y_train)

params = gpr.kernel_.get_params()
print(params)
