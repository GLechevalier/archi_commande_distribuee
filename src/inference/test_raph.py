import numpy as np
from scipy.optimize import curve_fit

class LawEstimator_data:
    def __init__(self):
        self.L = None
        self.data = None
        self.x0 = None
        self.y0 = None
        

    def update(self, pos, particle_intensity_value):
        """Ajoute de nouvelles données (positions et intensités)."""
        if self.L is None:
            self.L = pos.T
        else:
            self.L = np.concatenate((self.L, pos.T), axis=1)
            
        if self.data is None:
            self.data = particle_intensity_value
        else:
            self.data = np.concatenate((self.data, particle_intensity_value), axis=1)
    
    def declare_center(self, x0, y0):
        self.x0 = x0  # Valeur fixe de x0
        self.y0 = y0  # Valeur fixe de y0

    def twoD_Gaussian_fixed(self, coordinates, amplitude, sigma_x, sigma_y, theta):
        """Version de la fonction gaussienne où x0 et y0 sont fixes."""
        x, y = coordinates
        # Rotation et calcul des coefficients
        a = (np.cos(theta) ** 2) / (2 * sigma_x**2) + (np.sin(theta) ** 2) / (2 * sigma_y**2)
        b = -np.sin(2 * theta) / (4 * sigma_x**2) + np.sin(2 * theta) / (4 * sigma_y**2)
        c = (np.sin(theta) ** 2) / (2 * sigma_x**2) + (np.cos(theta) ** 2) / (2 * sigma_y**2)

        # Fonction gaussienne avec x0 et y0 fixes
        g = 310. + np.log10(
            amplitude * np.exp(
                -(a * ((x - self.x0) ** 2) + 2 * b * (x - self.x0) * (y - self.y0) + c * ((y - self.y0) ** 2))
            )
        )
        return g.ravel()

    def fit(self):
        """Effectue l'ajustement en gardant x0 et y0 fixes."""
        if self.L is None or self.data is None or self.L.shape[1] < 10:
            print("Pas assez de données pour ajuster.")
            return None

        if self.x0 is None or self.y0 is None:
            print("x0 et y0 doivent être définis avant d'effectuer le fit.")
            return None

        # Valeurs initiales pour les paramètres restants
        initial_guess = (1, 1, 1, 0)  # (amplitude, sigma_x, sigma_y, theta, offset)

        try:
            popt, _ = curve_fit(self.twoD_Gaussian_fixed, self.L, self.data.ravel(), p0=initial_guess)

            # Création du dictionnaire des paramètres optimisés
            popt_dict = {
                "amplitude": popt[0],
                "xo": self.x0,  # Fixé
                "yo": self.y0,  # Fixé
                "sigma_x": popt[1],
                "sigma_y": popt[2],
                "theta": popt[3],
            }
            self.popt_dict = popt_dict
            print("Fitted parameters:", popt_dict)
            return popt_dict
        except Exception as e:
            print(f"Erreur lors du fit: {e}")
            return None
