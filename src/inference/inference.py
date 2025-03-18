import matplotlib.pyplot as plt
import numpy as np
from ..lib.potential import Potential
from scipy.optimize import curve_fit


class DataGenerator:
    def __init__(self, distribution: Potential, n=100):
        self.pot = distribution
        x_moy, y_moy, sigma_1, sigma_2 = self.get_true_parameters()
        self.xmoy = x_moy
        self.ymoy = y_moy
        self.sigma_1 = sigma_1
        self.sigma_2 = sigma_2
        self.n = n
        return

    def get_true_parameters(self):
        pot = self.pot
        moy_1 = pot.mu1
        x_moy = moy_1[0]
        y_moy = moy_1[1]
        sigma_1 = pot.gaussian1.cov[0, 0]
        sigma_2 = pot.gaussian1.cov[1, 1]
        return x_moy, y_moy, sigma_1, sigma_2

    def generate_n_points(self):
        n = self.n
        L = []
        for i in range(n):
            L.append(
                [
                    np.random.uniform(self.pot.xmin, self.pot.xmax),
                    np.random.uniform(self.pot.ymin, self.pot.ymax),
                ]
            )
        return L

    def evaluate_n_points(self):
        n = self.n

        return_list = []
        L = self.generate_n_points()
        for i in range(n):
            return_list.append(self.pot.value(L[i]))
        return L, return_list

    def evaluate_n_points_from_list(self, L):
        return_list = []
        for i in range(len(L)):
            return_list.append(self.pot.value(L[i]))
        return return_list

    def generate_n_points_noise(self):
        L, Ly = self.evaluate_n_points()
        for i in range(len(Ly)):
            Ly[i] += np.random.normal()
        return L, Ly

    def gen_data(self):
        L, Ly_noise = self.generate_n_points_noise()
        L = np.array(L)
        Ly_true = self.evaluate_n_points_from_list(L)
        Ly_true = np.array(Ly_true)
        return L, Ly_true


class LawEstimator:
    def __init__(self, data_generator: DataGenerator):
        self.data_generator = data_generator
        self.L, self.Ly_true = self.data_generator.gen_data()
        return

    def diff(self, L, Ly_noise):
        Ly_true = self.data_generator.evaluate_n_points_from_list(L)
        sum = 0
        for i in range(len(L)):
            sum += (Ly_noise[i] - Ly_true[i]) ** 2
        return sum

    def twoD_Gaussian(
        self,
        coordinates,
        amplitude,
        xo,
        yo,
        sigma_x,
        sigma_y,
        theta,
        offset,
    ):
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
                -(
                    a * ((x - xo) ** 2)
                    + 2 * b * (x - xo) * (y - yo)
                    + c * ((y - yo) ** 2)
                )
            )
        )
        return g.ravel()  # Flatten the 2D array to 1D for curve_fit

    def fit(self, L=None, data=None):
        if L is None:
            L = self.L.transpose()
            print(L.shape)
        if data is None:
            data = self.Ly_true[:].reshape(1, len(self.Ly_true))
            print(data.shape)
        initial_guess = (1, 0, 0, 1, 1, 0, 0)

        popt, pcov = curve_fit(self.twoD_Gaussian, L, data.ravel(), p0=initial_guess)

        popt_dict = {
            "amplitude": popt[0],
            "xo": popt[1],
            "yo": popt[2],
            "sigma_x": popt[3],
            "sigma_y": popt[4],
            "theta": popt[5],
            "offset": popt[6],
        }
        self.popt_dict = popt_dict
        print("Fitted parameters:", popt_dict)
        return self.popt_dict


def main():
    pot1 = Potential(difficulty=1, random=False)
    fig2, ax2 = pot1.plot(2)

    dg = DataGenerator(pot1, n=100)

    le = LawEstimator(data_generator=dg)
    le.fit()
    return


if __name__ == "__main__":
    main()
