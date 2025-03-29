import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

class LawEstimator_data:
    def __init__(self):
        self.L = None  # Données filtrées utilisées pour le fit
        self.data = None  # Intensités filtrées
        self.L_original = None  # Sauvegarde des données brutes
        self.data_original = None  # Sauvegarde des intensités brutes
        self.x0 = None
        self.y0 = None
        self.popt_dict = None  # Stocke les paramètres ajustés
        self.r_squared = None  # Score de qualité du fit
        
        self.verbose = False
        
        self.confirmed = []
    
    def declare_confirmed(self, amp, gauss):
        self.confirmed.append((amp, gauss))
        

    def update(self, pos, particle_intensity_value):
        """Ajoute de nouvelles données (positions et intensités)."""
        if self.L_original is None:
            self.L_original = pos.T  # (2, N)
        else:
            self.L_original = np.concatenate((self.L_original, pos.T), axis=1)

        if self.data_original is None:
            self.data_original = particle_intensity_value
        else:
            self.data_original = np.concatenate((self.data_original, particle_intensity_value), axis=1)

        # Mettre à jour L et data pour qu'ils soient identiques aux données brutes
        self.L = np.copy(self.L_original)
        self.data = np.copy(self.data_original)

    def declare_center(self, x0, y0):
        """Déclare le centre fixe de la gaussienne."""
        self.x0 = x0
        self.y0 = y0

    def filter_data(self, max_distance):
        """Filtre les données en ne conservant que celles à une distance <= max_distance du centre (x0, y0)."""
        if self.data_original is None or self.L_original is None:
            print("Aucune donnée à filtrer.")
            return

        if self.x0 is None or self.y0 is None:
            print("x0 et y0 doivent être définis avant le filtrage.")
            return

        # Calcul des distances par rapport au centre (x0, y0)
        distances = np.sqrt((self.L_original[0] - self.x0) ** 2 + (self.L_original[1] - self.y0) ** 2)

        # Masque des points respectant la contrainte de distance
        mask = distances <= max_distance

        # Mise à jour des données filtrées
        self.L = self.L_original[:, mask]
        self.data = self.data_original[:, mask]

        if self.verbose:
            print(f"{mask.sum()} points conservés après filtrage sur distance pour inférence.")

    def search_zones(self, eps=0.3, min_samples=2):
        """
        Recherche des zones où le ratio est < 0.98, forme des clusters et 
        récupère le point ayant la plus grande mesure par cluster.

        Arguments :
            eps : Distance maximale pour regrouper des points en cluster (DBSCAN).
            min_samples : Nombre minimal de points pour former un cluster.

        Retourne :
            Tuple[np.ndarray, np.ndarray] :
                - Une liste de np.array([x, y]) pour les points sélectionnés.
                - Un tableau (N,) contenant les valeurs associées à ces points.
        """

        # Vérification des données
        if self.L_original is None or self.data_original is None or self.L_original.shape[1] < 1:
            print("Pas de données disponibles pour la recherche de zones.")
            return []

        # Extraction des coordonnées et valeurs mesurées
        x_data, y_data = self.L_original[0, :], self.L_original[1, :]
        measures = self.data_original.ravel()

        # Calcul de value pour chaque (x, y)
        computed_values = np.zeros_like(measures)
        for i, (x, y) in enumerate(zip(x_data, y_data)):
            value = 310. + np.log10(sum(source_conf[0] * source_conf[1].pdf((x, y)) for source_conf in self.confirmed) + 1e-309)
            computed_values[i] = value

        # Éviter les divisions par zéro en ajoutant un epsilon
        ratio = computed_values / measures

        # Filtrer les points avec ratio < 0.98
        mask = ratio < 0.98
        filtered_x = x_data[mask]
        filtered_y = y_data[mask]
        filtered_measures = measures[mask]

        if len(filtered_x) == 0:
            print("Aucune zone détectée avec un ratio < 0.98.")
            return []

        # Formation des clusters avec DBSCAN
        points = np.column_stack((filtered_x, filtered_y))
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)

        # Identifier le point avec la plus grande mesure par cluster
        cluster_labels = clustering.labels_
        unique_clusters = set(cluster_labels) - {-1}  # Exclure le bruit (-1)

        representative_points = []
        representative_values = []

        for cluster in unique_clusters:
            cluster_mask = cluster_labels == cluster
            cluster_measures = filtered_measures[cluster_mask]
            cluster_points = points[cluster_mask]

            # Sélection du point ayant la mesure la plus grande
            best_index = np.argmax(cluster_measures)
            representative_points.append(np.array(cluster_points[best_index]))  # np.array([x, y])
            representative_values.append(float(cluster_measures[best_index]))  # float()

        return representative_points, representative_values

    def twoD_Gaussian_fixed(self, coordinates, x0_search, y0_search, amplitude, sigma_x, sigma_y, sigma_xy):
        """Gaussienne 2D avec x0 et y0 fixes."""
        x, y = coordinates.T  # Convertir en (N,)

        # Éviter les valeurs négatives ou nulles de sigma_x et sigma_y
        sigma_x = max(abs(sigma_x), 1e-6)
        sigma_y = max(abs(sigma_y), 1e-6)

        # Matrice de covariance
        cov_matrix = np.array([
            [sigma_x, sigma_xy],
            [sigma_xy, sigma_y]
        ])

        # Vérification : rendre la matrice définie positive
        if np.linalg.det(cov_matrix) <= 0:
            sigma_xy = 0  # Éliminer la corrélation problématique
            cov_matrix = np.array([[sigma_x, 0.], [0., sigma_y]])
            
        # Création de la gaussienne centrée sur (x0, y0)
        gaussian = multivariate_normal(mean=[x0_search, y0_search], cov=cov_matrix)

        # Calcul de la densité de probabilité
        pdf_values = gaussian.pdf(np.column_stack((x, y)))
        pre_value = amplitude * pdf_values
        
        value = pre_value
        for source_conf in self.confirmed:
            value += source_conf[0]*source_conf[1].pdf(np.column_stack((x, y)))
            
        # Éviter les erreurs de log10 en imposant une valeur minimale
        g = 310. + np.log10(value + 1e-309)        
        return g

    def fit(self, apply_filter=True, estimation_max=310., verbose = True):
        """Ajuste les paramètres en gardant x0 et y0 fixes, avec augmentation du nombre max d'itérations."""
        if self.L_original is None or self.data_original is None or self.L_original.shape[1] < 10:
            print("Pas assez de données pour ajuster.")
            return None

        if self.x0 is None or self.y0 is None:
            print("x0 et y0 doivent être définis avant d'effectuer le fit.")
            return None

        # Valeurs initiales
        initial_guess = (self.x0, self.y0, np.power(10, estimation_max - 310.), 1, 1, 0)
        bounds = ([self.x0 - 0.1, self.y0 - 0.1, 0, 1e-6, 1e-6, 0], [self.x0 + 0.1, self.y0 + 0.1, np.inf, 3, 3, 3])

        continuer = True
        max_distance = 1
        r_save = -1
        while continuer:
            
            # Appliquer le filtrage par distance si demandé
            if apply_filter and max_distance is not None:
                self.filter_data(max_distance)

            # Transformation des données pour curve_fit
            x_data = self.L.T  # (N, 2)
            y_data = self.data.ravel()  # (N,)
            
            try:
                popt, _ = curve_fit(
                    self.twoD_Gaussian_fixed,
                    x_data,
                    y_data,
                    p0=initial_guess,
                    bounds=bounds,
                    maxfev=5000
                )

                
                # Calcul du score R²
                y_pred = self.twoD_Gaussian_fixed(x_data, *popt)
                ss_total = np.sum((y_data - np.mean(y_data)) ** 2)
                ss_residual = np.sum((y_data - y_pred) ** 2)
                self.r_squared = 1 - (ss_residual / ss_total)

            except RuntimeError:
                print("Échec de l'optimisation : essaye avec des valeurs initiales différentes.")
                self.r_squared = -1
            
            max_distance += 0.5
            if (r_save > self.r_squared) or (max_distance > 25):
                self.r_squared = r_save
                continuer = False
                
            elif (self.r_squared >= 0.99): # and (max_distance > 5):
                self.x0 = popt[0]
                self.y0 = popt[1]
                
                # Stockage des paramètres optimisés
                self.popt_dict = {
                    "amplitude": popt[2],
                    "xo": popt[0],
                    "yo": popt[1],
                    "sigma_x": popt[3],
                    "sigma_y": popt[4],
                    "sigma_xy": popt[5],
                }
            
                continuer = False
                
            else:
                r_save = self.r_squared
                
                if self.r_squared != -1:
                    # Stockage des paramètres optimisés
                    self.x0 = popt[0]
                    self.y0 = popt[1]
                    
                    self.popt_dict = {
                        "amplitude": popt[2],
                        "xo": popt[0],
                        "yo": popt[1],
                        "sigma_x": popt[3],
                        "sigma_y": popt[4],
                        "sigma_xy": popt[5],
                    }
                
                    initial_guess = (popt[0], popt[1], popt[2], abs(popt[3]), abs(popt[4]), popt[5])
                    bounds = ([popt[0] - 0.1, popt[1] - 0.1, 0, 1e-6, 1e-6, 0.], [popt[0] + 0.1, popt[1] + 0.1, np.inf, 3, 3, 3])
                    
                else:
                    # Stockage des paramètres optimisés
                    self.popt_dict = {
                        "amplitude": initial_guess[2],
                        "xo": self.x0,
                        "yo": self.y0,
                        "sigma_x": 1,
                        "sigma_y": 1,
                        "sigma_xy": 0,
                    }
                    
        if verbose:     
            print("Rayon d'inférence :", max_distance)
            print("Paramètres ajustés :", self.popt_dict)
            print(f"Score R² pred: {self.r_squared:.4f}")
            self.plot_fit()
            
        # Calcul du score R²
        x_data = np.copy(self.L_original).T  # (N, 2)
        y_data = np.copy(self.data_original).ravel()  # (N,)
        y_pred = self.twoD_Gaussian_fixed(x_data, *popt)
        ss_total = np.sum((y_data - np.mean(y_data)) ** 2)
        ss_residual = np.sum((y_data - y_pred) ** 2)
        r_tot = 1 - (ss_residual / ss_total)
        
        print(f"Score R² total: {r_tot:.4f}")
            
        
    
    def get_solution(self):
        
        # Matrice de covariance
        cov_matrix = np.array([[self.popt_dict["sigma_x"], self.popt_dict["sigma_xy"]], [self.popt_dict["sigma_xy"], self.popt_dict["sigma_y"]]])

        # Vérification : rendre la matrice définie positive
        if np.linalg.det(cov_matrix) <= 0:
            cov_matrix = np.array([[self.popt_dict["sigma_x"], 0.], [0., self.popt_dict["sigma_y"]]])
            
        gaussian_sol = multivariate_normal([self.x0, self.y0], cov_matrix)
        
        return np.array([self.x0, self.y0]), self.popt_dict["amplitude"], gaussian_sol, self.r_squared
    
      
    def plot_fit(self, colorbar=True):
        """Affiche la gaussienne ajustée en utilisant les mêmes méthodes que la génération du champ potentiel."""

        if not hasattr(self, "popt_dict"):
            print("Aucun fit trouvé. Exécute d'abord `fit()`.")
            return

       # Définition des bornes de l'espace de visualisation
        self.xmin, self.xmax, self.xstep = -25, 25, 0.5
        self.ymin, self.ymax, self.ystep = -25, 25, 0.5

        # Grille des points
        x, y = np.mgrid[self.xmin:self.xmax:self.xstep, self.ymin:self.ymax:self.ystep]
        pos = np.dstack((x, y))
        
        # Matrice de covariance
        cov_matrix = np.array([[self.popt_dict["sigma_x"], self.popt_dict["sigma_xy"]], [self.popt_dict["sigma_xy"], self.popt_dict["sigma_y"]]])

        # Vérification : rendre la matrice définie positive
        if np.linalg.det(cov_matrix) <= 0:
            cov_matrix = np.array([[self.popt_dict["sigma_x"], 0.], [0., self.popt_dict["sigma_y"]]])
            
        gaussian_sol = multivariate_normal([self.x0, self.y0], cov_matrix)
        
        # Remplacement de `self.value(pos)` par l'utilisation de la fonction ajustée
        potentialFieldForPlot = 310. + np.log10(self.popt_dict["amplitude"]*gaussian_sol.pdf(pos) + 1e-309)

        # Création de la figure si nécessaire
        fig = plt.figure(30)
        fig.clf()
        ax = fig.add_subplot(111)

        # Contour du champ potentiel ajusté
        cs = ax.contourf(x, y, potentialFieldForPlot, 20, cmap='BrBG')

        # Ajout des points de données
        if hasattr(self, "L"):
            x_values, y_values = self.L
            ax.scatter(x_values, y_values, color="red", s=5, label="Données utilisées")

        # Affichage du centre fixé
        ax.scatter(self.x0, self.y0, c="white", marker="x", s=100, label="Centre fixé")

        # Ajout d'une barre de couleur si demandé
        if colorbar:
            fig.colorbar(cs)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.legend()
        ax.set_title(f"Fit Gaussien 2D (R²={self.r_squared:.4f})")

        plt.pause(0.1)