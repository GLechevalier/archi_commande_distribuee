# -*- coding: utf-8 -*-
"""
author: Sylvain Bertrand, 2025

   All variables are in SI units
    
   
   Variables used by the functions of this script
    - t: time instant (s)
    - robotNo: no of the current robot for which control is coputed (0 .. nbRobots-1)
    - poses:  size (3 x nbRobots)
        eg. of use: the pose of robot 'robotNo' can be obtained by: poses[:,robotNo]
            poses[robotNo,0]: x-coordinate of robot position (in m)
            poses[robotNo,1]: y-coordinate of robot position (in m)
            poses[robotNo,2]: orientation angle of robot (in rad)   (in case of unicycle dynamics only)
"""


import numpy as np
import math
import matplotlib.pyplot as plt
from lib.potential import Potential
from lib.gridmap import gridmap
from inference.test_raph import LawEstimator_data


# ==============   "GLOBAL" VARIABLES KNOWN BY ALL THE FUNCTIONS ==============
# all variables declared here will be known by functions below
# use keyword "global" inside a function if the variable needs to be modified by the function

# global toto

global firstCall   # can be used to check the first call ever of a function
firstCall = True

global pot # DO NOT MODIFY - allows initialisation of potential function from this script
global detected_sources
detected_sources = []

global gridmap_record
global law_est

global phase
phase = 1

global radius
radius = 1

# =============================================================================


# =============================================================================
def potential_seeking_ctrl(t, robotNo, robots_poses):
# =============================================================================

    # --- example of modification of global variables ---
    # ---(updated values of global variables will be known at next call of this funtion) ---
    # global toto
    # toto = toto +1
    
    global firstCall
    global pot
    global detected_sources
    global gridmap_record
    global law_est
    global phase
    


    # --- part to be run only once --- 
    if firstCall:
    
        # !!!!!!!!!!!!!!!!!!!!!!!  DO NOT REMOVE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #     YOU CAN MODIFY difficulty {1,2,3} AND random {True, False} PARAMETERS
        pot = Potential(difficulty=2, random=False) 
        pot.get_truth() 
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        gridmap_record = gridmap(-25, 25, -25, 25, 0.5, 0.5) # A bit dangerous to do so because not
        law_est = LawEstimator_data()
        # you can add here other instructions to be executed only once
        
        firstCall = False
    # --------------------------------
    
    
    # get number of robots  (short notation)
    N = robots_poses.shape[0]
    
    # get index of current robot  (short notation)
    i = robotNo

    # get positions of all robots (short notation)
    x = robots_poses[:,0:2]

    # get potential values measured by all robots at their current positions (short notation)
    pot_measurement = np.zeros(N)
    for m in range(N):
        pot_measurement[m] = pot.value(x[m,:])
        #gridmap_record.update(x[m, 0], x[m, 1], pot_measurement[m], t)
        

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # to get access to potential measurement from robot i at time t in the rest of the code
    # you can use eihter use    pot_measurement[i]     or      pot.value(x[i,:])
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
    law_est.update(x[robotNo:robotNo+1, :], np.array([[pot_measurement[robotNo]]]))
    if phase ==  1:
        ui = formation_gradient(t, robotNo, N, x, pot_measurement)
    elif phase == 2:
        ui = formation_recherche(t, robotNo, N, x, pot_measurement)
    
    return ui[0], ui[1], pot   # potential is also returned to be used by main script for displays (DO NOT MODIFY)
# =============================================================================

def get_gridmap():
    global gridmap_record
    
    return gridmap_record

def plot_fit(estimator, xmin, xmax, ymin, ymax, xstep=0.1, ystep=0.1, fig=None, ax=None, noFigure=None, colorbar=True):
    """
    Affiche le champ de potentiel estimé à partir du fit d'une gaussienne.

    :param estimator: Instance de `LawEstimator_data` avec les paramètres ajustés.
    :param xmin, xmax, ymin, ymax: Limites du graphique.
    :param xstep, ystep: Pas de discrétisation pour le champ de potentiel.
    :param fig, ax: Objet figure et axe matplotlib (facultatif).
    :param noFigure: Numéro de la figure si elle est créée.
    :param colorbar: Affiche la barre de couleur si True.
    """
    
    if not hasattr(estimator, "popt_dict"):
        print("Erreur : Aucun fit trouvé. Exécutez `fit()` d'abord.")
        return

    # Récupération des paramètres ajustés
    popt = estimator.popt_dict
    amplitude, sigma_x, sigma_y, theta = popt["amplitude"], popt["sigma_x"], popt["sigma_y"], popt["theta"]
    x0, y0 = popt["xo"], popt["yo"]

    # Création de la grille de points
    x, y = np.mgrid[xmin:xmax:xstep, ymin:ymax:ystep]
    pos = np.dstack((x, y))
    
    # Calcul du potentiel estimé sur la grille
    def gaussian_value(coords):
        """Calcule la valeur de la gaussienne aux coordonnées données."""
        x, y = coords
        a = (np.cos(theta) ** 2) / (2 * sigma_x**2) + (np.sin(theta) ** 2) / (2 * sigma_y**2)
        b = -np.sin(2 * theta) / (4 * sigma_x**2) + np.sin(2 * theta) / (4 * sigma_y**2)
        c = (np.sin(theta) ** 2) / (2 * sigma_x**2) + (np.cos(theta) ** 2) / (2 * sigma_y**2)
        
        return 310. + np.log10(
            amplitude * np.exp(-(
                a * ((x - x0) ** 2) + 2 * b * (x - x0) * (y - y0) + c * ((y - y0) ** 2)
            ))
        )

    potentialFieldForPlot = gaussian_value((x, y))

    # Création de la figure
    if fig is None:
        if noFigure is None:
            noFigure = 1
        fig = plt.figure(noFigure)
    if ax is None:
        ax = fig.add_subplot(111)

    # Séparation des coordonnées x et y des données
    x_values, y_values = estimator.L  # L contient les coordonnées des points (déjà transposées)

    # Affichage du champ de potentiel estimé
    cs = ax.contourf(x, y, potentialFieldForPlot, 20, cmap="BrBG")

    # Ajout des points de données réels
    ax.scatter(x_values, y_values, color="red", edgecolors="black", label="Données réelles")

    # Ajout de la barre de couleur
    if colorbar:
        fig.colorbar(cs)

    # Ajout du titre et labels
    ax.set_title("Ajustement Gaussien")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend()

    plt.show()


# general template of a function defining a control law
# =============================================================================
def formation_gradient(t, robotNo, N, x, measurement):
# ============================================================================= 
    
    global detected_sources
    global law_est
    global phase

    formation_distance = 1
    relative_pose = np.array([[formation_distance*np.sin(2*np.pi*i/N) for i in range(N)],       # x-coordinates (m)
                                [formation_distance*np.cos(2*np.pi*i/N) for i in range(N)]]).T   # y-coordinates (m)
    
    sum_distances = 0.
    vel_vector = np.zeros(2)
    grad_total  = np.zeros(2)
    
    for i in range(N):
        if i != robotNo:
            dist = np.linalg.norm(x[robotNo, :] - x[i, :])
            dist_rel = np.linalg.norm(relative_pose[robotNo, :] - relative_pose[i, :])
            sum_distances += max(dist - dist_rel, 0)
            vel_vector += ((dist - dist_rel)/dist)*(x[i, :] - x[robotNo, :])
            
        if measurement[robotNo] == -10: ## Correction in order to go toward the center of the field if all the robots are too far from sources
            dist = np.linalg.norm(x[robotNo, :] - np.array([0, 0, 0]))
            vel_vector += (1/dist)*(np.array([0, 0, 0] - x[robotNo, :]))

    for i in range(N):
        for j in range(N):
            if i != j:
                grad_local = (measurement[i] - measurement[j]) / np.linalg.norm(x[i, :] - x[j, :])
                grad_total +=  np.array([grad_local*(x[j, 0] - x[i, 0]), grad_local*(x[j, 1] - x[i, 1])])
    
    # initialize control input vector for current robot i
    kp = 1.5
    kg = 5
    kgi = 0.3
    kto = 0.1
    pi = 0.1
    dist_consensus = np.linalg.norm(vel_vector)
    factor_consensus = kp*np.arctan(dist_consensus)/dist_consensus
    
    dist_grad = np.linalg.norm(grad_total)
    if dist_grad < 0.0001:
        factor_grad = 0
    else:
        factor_grad = (kg*np.arctan(kgi*dist_grad)/dist_grad)*(np.pi - 2*np.arctan(sum_distances*kto))/(np.pi)
    
    v = (factor_consensus*vel_vector - factor_grad*grad_total)*(np.pi - 2*np.arctan((measurement[robotNo] - min(measurement))*pi))/(np.pi)
    if t == 10.:
        print(sum_distances, dist_grad)
    
    #print(dist_grad)
    if (sum_distances < 1.) and (dist_grad < 0.05):
        potential_source = np.array([np.mean(x[:, 0]), np.mean(x[:, 1])])
        for source in detected_sources:
            if np.linalg.norm(source - potential_source) < 0.1:
                break
        
        else:
            law_est.declare_center(np.mean(x[:, 0]), np.mean(x[:, 1]))
            law_est.fit()
            #plot_fit(law_est, xmin=-25, xmax=25, ymin=-25, ymax=25, xstep=0.1, ystep=0.1)
            detected_sources.append(potential_source)
            print("source détéctée", potential_source)
            #phase = 2

    # .................  TO BE COMPLETED HERE .............................
    return v
# =============================================================================

# general template of a function defining a control law
# =============================================================================
def formation_recherche(t, robotNo, N, x, measurement):
# ============================================================================= 
    
    global detected_sources
    global law_est
    global radius
    
    radius += 0.1
    formation_distance = radius
    relative_pose = np.array([[formation_distance*np.sin(2*np.pi*i/N) for i in range(N)],       # x-coordinates (m)
                                [formation_distance*np.cos(2*np.pi*i/N) for i in range(N)]]).T   # y-coordinates (m)
    
    sum_distances = 0.
    vel_vector = np.zeros(2)
    grad_total  = np.zeros(2)
    
    for i in range(N):
        if i != robotNo:
            dist = np.linalg.norm(x[robotNo, :] - x[i, :])
            dist_rel = np.linalg.norm(relative_pose[robotNo, :] - relative_pose[i, :])
            sum_distances += max(dist - dist_rel/radius, 0)
            vel_vector += ((dist - dist_rel)/dist)*(x[i, :] - x[robotNo, :])
            
        if measurement[robotNo] == -10: ## Correction in order to go toward the center of the field if all the robots are too far from sources
            dist = np.linalg.norm(x[robotNo, :] - np.array([0, 0, 0]))
            vel_vector += (1/dist)*(np.array([0, 0, 0] - x[robotNo, :]))

    dist = np.linalg.norm(x[robotNo, :] - x[(robotNo + 1)%N, :])
    vel_vector += (100/dist)*(x[(robotNo + 1)%N, :] - x[robotNo, :])
    
    for i in range(N):
        for j in range(N):
            if i != j:
                grad_local = (measurement[i] - measurement[j]) / np.linalg.norm(x[i, :] - x[j, :])
                grad_total +=  np.array([grad_local*(x[j, 0] - x[i, 0]), grad_local*(x[j, 1] - x[i, 1])])
    
    # initialize control input vector for current robot i
    kp = 1.5
    kg = 5
    kgi = 0.3
    kto = 0.1
    dist_consensus = np.linalg.norm(vel_vector)
    factor_consensus = kp*np.arctan(dist_consensus)/dist_consensus
    
    dist_grad = np.linalg.norm(grad_total)
    if dist_grad < 0.0001:
        factor_grad = 0
    else:
        factor_grad = (kg*np.arctan(kgi*dist_grad)/dist_grad)*(np.pi - 2*np.arctan(sum_distances*kto))/(np.pi)
    
    v = factor_consensus*vel_vector - factor_grad*grad_total
    if t == 10.:
        print(sum_distances, dist_grad)

    # .................  TO BE COMPLETED HERE .............................
    return v
# =============================================================================

