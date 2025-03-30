import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

class LawEstimator_data:
    def __init__(self, N):
        self.N = N
        
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
        
        self.x0 = None
        self.y0 = None
        self.popt_dict = None  
        self.r_squared = None
        

    def update(self, pos, particle_intensity_value, max_data_size=10000):
        """Ajoute de nouvelles données et applique un nettoyage périodique si besoin.
        
        - Si les données dépassent max_data_size, on applique un filtre :
        * On supprime les N premiers points,
        * Puis on garde les N suivants, et ainsi de suite.
        
        Arguments :
            pos : np.array (2, M) - Positions des nouvelles mesures.
            particle_intensity_value : np.array (1, M) - Valeurs d'intensité des particules.
            max_data_size : int - Nombre maximum de points avant déclenchement du nettoyage.
        """
        
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

        # Vérifier si le nombre de points dépasse la limite
        if self.L.shape[1] > max_data_size:
            print(f"Nettoyage des données : {self.L.shape[1]} points -> réduction.")

            # Création d'une liste d'indices à conserver selon le schéma [Supprimer N, garder N]
            indices_to_keep = np.array([
                i for i in range(self.L.shape[1]) 
                if (i // self.N) % 2 == 1  # Supprime les N premiers, garde les N suivants
            ])

            # Appliquer le filtrage
            self.L_original = self.L_original[:, indices_to_keep]
            self.data_original = self.data_original[:, indices_to_keep]

            # Mettre à jour les données finales
            self.L = np.copy(self.L_original)
            self.data = np.copy(self.data_original)

            print(f"Données réduites à {self.L.shape[1]} points.")

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

        self.n_don = mask.sum()
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
            return [], []

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
            return [], []

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
        if np.linalg.det(cov_matrix) <= 1e-4:
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

    def fit(self, apply_filter=True, estimation_max=310., pre_estimation=None, verbose = True):
        """Ajuste les paramètres en gardant x0 et y0 fixes, avec augmentation du nombre max d'itérations."""
        if self.L_original is None or self.data_original is None or self.L_original.shape[1] < 10:
            print("Pas assez de données pour ajuster.")
            return None

        if self.x0 is None or self.y0 is None:
            print("x0 et y0 doivent être définis avant d'effectuer le fit.")
            return None

        # Valeurs initiales
        if pre_estimation:
            initial_guess = (self.x0, self.y0, pre_estimation, 1, 1, 0)
        else:
            initial_guess = (self.x0, self.y0, np.power(10, estimation_max - 310.), 1, 1, 0)
            
        bounds = ([self.x0 - 0.1, self.y0 - 0.1, 0, 1e-2, 1e-2, 0], [self.x0 + 0.1, self.y0 + 0.1, np.inf, 3, 3, 3])
        self.n_don = 0

        continuer = True
        max_distance = 1
        r_save = -10
        dis_save = 1
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
                    maxfev=1000
                )

                
                # Calcul du score R²
                y_pred = self.twoD_Gaussian_fixed(x_data, *popt)
                ss_total = np.sum((y_data - np.mean(y_data)) ** 2)
                ss_residual = np.sum((y_data - y_pred) ** 2)
                self.r_squared = 1 - (ss_residual / ss_total)

            except RuntimeError:
                print("Échec de l'optimisation : essaye avec des valeurs initiales différentes.")
                self.r_squared = -10
            
            if ((r_save > self.r_squared) and (max_distance >= 2)) or (max_distance > 15):
                self.r_squared = r_save
                continuer = False
                
            elif (self.r_squared >= 0.99) and (max_distance >= 2):
                dis_save = max_distance
                
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
                
            elif (r_save > self.r_squared):
                # Résolution d'un problème de sur-confiance en peu de données
                if self.n_don < 50:
                    r_save = self.r_squared - 0.2
                elif (self.n_don < 150) or (max_distance < 2):
                    r_save = self.r_squared - 0.05
                else:
                    r_save = self.r_squared

                dis_save = max_distance
                
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
                bounds = ([popt[0] - 0.1, popt[1] - 0.1, 0, 1e-2, 1e-2, 0.], [popt[0] + 0.1, popt[1] + 0.1, np.inf, 3, 3, 3])
                
            elif (r_save == -10):
                # Stockage des paramètres optimisés
                self.popt_dict = {
                    "amplitude": initial_guess[2],
                    "xo": self.x0,
                    "yo": self.y0,
                    "sigma_x": 1,
                    "sigma_y": 1,
                    "sigma_xy": 0,
                }
                
            max_distance += 0.5
                    
        if verbose:     
            print("Rayon d'inférence :", dis_save)
            print("Paramètres ajustés :", self.popt_dict)
            print(f"Score R² pred: {self.r_squared:.4f}")
            
            #self.plot_fit()
            
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
        
        return np.array([self.x0, self.y0]), self.popt_dict["amplitude"], gaussian_sol, self.r_squared, cov_matrix
    
      
    def plot_fit(self, colorbar = True, fit_actuel = True, save = None):
        """Affiche deux subplots :
        - 1er subplot : Solution globale avec toutes les sources confirmées.
        - 2e subplot : Source détectée actuellement.
        """

        if not hasattr(self, "popt_dict") or (self.popt_dict is None):
            fit_actuel = False
            center_fit = None
            total_field = 0
        else:
            center_fit, amp_fit, gauss_fit, _, _ = self.get_solution()
            total_field = amp_fit*gauss_fit.pdf(pos)

        # Définition des bornes de l'espace de visualisation
        self.xmin, self.xmax, self.xstep = -25, 25, 0.5
        self.ymin, self.ymax, self.ystep = -25, 25, 0.5

        # Grille des points
        x, y = np.mgrid[self.xmin:self.xmax:self.xstep, self.ymin:self.ymax:self.ystep]
        pos = np.dstack((x, y))

        if fit_actuel:
            # Création de la figure et des subplots
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            ax0 = axes[0]
        
        else:
            fig, ax0 = plt.subplots(1, 1, figsize=(6, 6))
        
        
        # === 1er SUBPLOT : SOLUTION TOTALE AVEC TOUTES LES SOURCES ===
        

        for source_conf in self.confirmed:
            amplitude, source_gaussian = source_conf
            total_field += amplitude * source_gaussian.pdf(pos)

        # Transformation logarithmique
        potential_total = 310. + np.log10(total_field + 1e-309)
        
        cs1 = ax0.contourf(x, y, potential_total, 20, cmap='BrBG')

        # Affichage des centres confirmés
        flag = True
        for source_conf in self.confirmed:
            _, source_gaussian = source_conf
            mean_x, mean_y = source_gaussian.mean
            if flag:
                flag = False
                ax0.scatter(mean_x, mean_y, c="black", marker="x", s=100, label="Sources confirmées")
            else:
                ax0.scatter(mean_x, mean_y, c="black", marker="x", s=100)
                
        if center_fit is not None:
            ax0.scatter(center_fit[0], center_fit[1], c="blue", label="Source détéctée")
                
        ax0.set_xlabel("X")
        ax0.set_ylabel("Y")
        ax0.set_title("Estimation globale")
        ax0.legend()

        if colorbar:
            fig.colorbar(cs1, ax=ax0)

        if fit_actuel:
            # === 2e SUBPLOT : SOURCE ACTUELLE & SOURCES CONFIRMÉES INDIVIDUELLEMENT ===
            axes[1].set_title("Source Actuelle")
            
            current_field = 310. + np.log10(amp_fit*gauss_fit.pdf(pos) + 1e-309)

            # Affichage de la source actuelle
            cs2 = axes[1].contourf(x, y, current_field, 20, cmap='BrBG')
                
            # Ajout des points de données
            if hasattr(self, "L"):
                x_values, y_values = self.L
                axes[1].scatter(x_values, y_values, color="red", s=5, label="Données utilisées")

            axes[1].set_xlabel("X")
            axes[1].set_ylabel("Y")
            axes[1].legend()

            if colorbar:
                fig.colorbar(cs2, ax=axes[1])

        # Ajustement de l'affichage
        plt.tight_layout()
        plt.pause(0.1)
        
        if save:
            plt.savefig("Results/"+save+"_Estimation.png")