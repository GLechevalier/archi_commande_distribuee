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
firstCall = False

global pot # DO NOT MODIFY - allows initialisation of potential function from this script
global controleur

class Controleur():
    def __init__(self, N, pot, params, limits, fenetre:int):
        self.phase = 1
        self.pot = pot
        
        self.centers = [None]*fenetre
        self.detected_sources = {}
        
        self.gridmap_record = gridmap(limits[0], limits[1], limits[2], limits[3], limits[4], limits[5])
        self.radius = 1
        
        self.law_est = LawEstimator_data(N)
        
        self.kp = params[0]
        self.kg = params[1]
        self.kpi = params[2]
        self.kgi = params[3]
        self.kto = params[4]
        
        self.kr = min(0.99, params[5])/(1 - min(0.99, params[5]))
        
        self.kp_r = params[5]
        self.kpi_r = params[6]
        self.krep = params[7]
        
        self.t = -1
        self.source_verif = 0
        self.confirmed = False
        self.mesured = False
        
        self.zones_recherche = None
        
        
    def prev(self, pos):
        value = 0
        for source in self.detected_sources.values():
            value += source[1]*source[2].pdf(pos)
        return 310. + np.log10(value + 1e-309)
    
    def commands(self, t, robotNo, N, x, measurement):
        
        self.law_est.update(x[robotNo:robotNo+1, :], np.array([[measurement[robotNo]]]))
        if self.phase ==  1:
            ui = self.formation_gradient(t, robotNo, N, x, measurement)
        elif self.phase == 2:
            ui = self.recherche_verifications(t, robotNo, N, x, measurement)
        elif self.phase == 3:
            ui = self.regroupement(t, robotNo, N, x, measurement)
            
        return ui
    
    def get_gridmap(self):
        return self.gridmap_record
    
    # general template of a function defining a control law
    # =============================================================================
    def formation_gradient(self, t, robotNo, N, x, measurement):
    # ============================================================================= 

        formation_distance = 1
        relative_pose = np.array([[formation_distance*np.sin(2*np.pi*i/N) for i in range(N)],       # x-coordinates (m)
                                    [formation_distance*np.cos(2*np.pi*i/N) for i in range(N)]]).T   # y-coordinates (m)
        
        sum_distances = 0.
        vel_vector = np.zeros(2)
        grad_total  = np.zeros(2)
        vector_rep =  np.zeros(2)
        
        for i in range(N):
            if i != robotNo:
                dist = np.linalg.norm(x[robotNo, :] - x[i, :])
                dist_rel = np.linalg.norm(relative_pose[robotNo, :] - relative_pose[i, :])
                sum_distances += max((dist - dist_rel)/self.radius, 0)
                vel_vector += ((dist - dist_rel)/dist)*(x[i, :] - x[robotNo, :])
                
            if measurement[robotNo] == -10: ## Correction in order to go toward the center of the field if all the robots are too far from sources
                dist = np.linalg.norm(x[robotNo, :] - np.array([0, 0]))
                vel_vector += (1/dist)*(np.array([0, 0]) - x[robotNo, :])

        # Calcul du gradient local à partir de la formation
        for i in range(N):
            for j in range(N):
                if i != j:
                    grad_local = (measurement[i] - measurement[j]) / np.linalg.norm(x[i, :] - x[j, :])
                    grad_total +=  np.array([grad_local*(x[j, 0] - x[i, 0]), grad_local*(x[j, 1] - x[i, 1])])
        
        
        
        # Calcul du poids du contrôle pour le consensus
        dist_consensus = np.linalg.norm(vel_vector)
        if dist_consensus < 1e-8:
            factor_consensus = 0
        else:
            factor_consensus = self.kp*np.arctan(self.kpi*dist_consensus)/dist_consensus
        
        # Cacul du poids du contrôle pour la descente de gradient
        dist_grad = np.linalg.norm(grad_total)
        if dist_grad < 1e-8:
            factor_grad = 0
        else:
            factor_grad = (self.kg*np.arctan(self.kgi*dist_grad)/dist_grad)*(np.pi - 2*np.arctan(self.kto*sum_distances))/(np.pi)
        
        # Calcul du vecteur de repoussement
        for source in self.detected_sources.values():
            vector_rep_tmp = x[robotNo, :] - source[0] # Le robot est repoussé par les sources déjà découvertes
            
            # Calcul du poids du contrôle pour l'évitement des zones déjà couvertes
            dist_rep_tmp = np.linalg.norm(vector_rep_tmp)
            if dist_rep_tmp < 1e-2:
                vector_rep += np.array([0, 1]) # Si on est sur la soource on doit bouger rapidement (n'arrive normalement jamais)
            else:
                ratio = 1 - (self.prev(x[robotNo, :]) / measurement[robotNo])
                factor_rep_tmp = (np.pi - 2*np.arctan(200*ratio))/(np.pi)
                
                vector_rep += factor_rep_tmp*vector_rep_tmp
        
        
        #print(dist_consensus, factor_consensus, dist_grad, factor_grad)
        v = (factor_consensus*vel_vector - factor_grad*grad_total + self.krep*vector_rep) #*(np.pi - 2*np.arctan((measurement[robotNo] - min(measurement))*pi))/(np.pi)
        
        # Regarde le mouvement de la formation entière à partir du mouvement du centre
        dist_mouv = 10
        center_formation = np.array([np.mean(x[:, 0]), np.mean(x[:, 1])])
        if self.t != t:            
            dist_mouv = 0
            for ct in self.centers:
                if ct is None:
                    dist_mouv = 10
                    break
                else:
                    dist_mouv += np.linalg.norm(ct - center_formation)
                    
            self.t = t
            self.centers.pop(0)
            self.centers.append(center_formation)
            
        if (sum_distances < 1.) and (dist_mouv < 0.05):
            for source in self.detected_sources.values():
                if np.linalg.norm(source[0] - center_formation) < 0.5: ## On repart sur une phase de recherche car on est retombé sur la même source
                    self.confirmed = True
                    self.mesured = True
                    self.first = True
                    self.cnt = 0
                    self.last = 0
                    
                    self.phase = 2
                    for (id, source_test) in self.detected_sources.items():
                        if np.linalg.norm(source[0] - source_test[0]) < 1e-3:
                            self.source_verif = id
                    self.radius = 1.
                    
                    self.zones_recherche = None

            
            else: # On a trouver une nouvelle source, on essaye d'estimer sa diffusion, puis on part à la recherche d'une nouvelle
                self.law_est.declare_center(np.mean(x[:, 0]), np.mean(x[:, 1]))
                self.law_est.fit(estimation_max=max(measurement))
                
                center, amp, gauss, r, cov = self.law_est.get_solution()
                #self.pot.plot_essai(amp, gauss)
                
                id = len(self.detected_sources) + 1
                self.detected_sources[id] = [center, amp, gauss, r, cov]
                print("source détéctée", center, " à t=", t)
                self.phase = 2
                self.source_verif = id
                self.radius = 1.
                    
                self.confirmed = False
                self.mesured = False
                self.first = True
                self.cnt = 0
                self.last = 0
                
                self.zones_recherche = None

        # .................  TO BE COMPLETED HERE .............................
        return v
    
    # general template of a function defining a control law
    # =============================================================================
    def recherche_verifications(self, t, robotNo, N, x, measurement):
    # ============================================================================= 
        
        formation_distance = self.radius
        relative_pose = np.array([[formation_distance*np.sin(2*np.pi*i/(N-1)) for i in range(N-1)] + [0],       # x-coordinates (m)
                                    [formation_distance*np.cos(2*np.pi*i/(N-1)) for i in range(N-1)] + [0]]).T   # y-coordinates (m)
        
        sum_distances = 0.
        vel_vector = np.zeros(2)
        
        if robotNo == N-1:
            
            vel_vector = self.detected_sources[self.source_verif][0] - x[robotNo, :]
            
            if np.linalg.norm(vel_vector) < 0.001:
                self.mesured = True

            if self.radius < 15.:
                # Adaptation de la vitesse de croissance
                rad = sum(np.linalg.norm(x[robotNo, :] - x[i, :]) for i in range(N-1))/(N-1)
                if (self.radius - rad) < 2.0:
                    self.radius += 0.3
                
                if not self.confirmed:
                    
                    if self.last > 5:
                        self.law_est.fit(pre_estimation = self.detected_sources[self.source_verif][1], verbose = False)
                        center, amp, gauss, r, cov = self.law_est.get_solution()
                        self.last = 0
                        
                        if r > self.detected_sources[self.source_verif][3]:
                            self.detected_sources[self.source_verif] = [center, amp, gauss, r, cov]
                            print("Sources affinées: ", ["id:" + str(id) + " amp-" + str(det_sources[1]) + " (x, y)-(" + str(det_sources[0][0]) + ", " + str(det_sources[0][1]) +
                                                         ") sigma- " + str(det_sources[4][0, 0]) + ", " + str(det_sources[4][1, 1]) + ", " + str(det_sources[4][0, 1])
                                                         for (id, det_sources) in self.detected_sources.items()])
                        
                        if self.detected_sources[self.source_verif][3] > 0.999 and self.mesured:
                            self.law_est.declare_confirmed(self.detected_sources[self.source_verif][1], self.detected_sources[self.source_verif][2])
                            self.confirmed = True
                    
                    else:
                        self.last += 1
                    
                    
            elif not self.confirmed:
                print("Confirmation: ", ["id:" + str(id) + " amp-" + str(det_sources[1]) + " (x, y)-(" + str(det_sources[0][0]) + ", " + str(det_sources[0][1]) +
                                                         ") sigma-" + str(det_sources[4][0, 0]) + ", " + str(det_sources[4][0, 1]) + ", " + str(det_sources[4][1, 1])
                                                         for (id, det_sources) in self.detected_sources.items()])
                center, amp, gauss, r, _ = self.law_est.get_solution()
                
                self.law_est.declare_confirmed(amp, gauss)
                self.confirmed = True

        else:
            for i in range(N):
                if i != robotNo:
                    dist = np.linalg.norm(x[robotNo, :] - x[i, :])
                    dist_rel = np.linalg.norm(relative_pose[robotNo, :] - relative_pose[i, :])
                    sum_distances += max((dist - dist_rel)/self.radius, 0)
                    vel_vector += ((dist - dist_rel)/dist)*(x[i, :] - x[robotNo, :])

            dist = np.linalg.norm(x[robotNo, :] - x[(robotNo + 1)%(N-1), :])
            norm = np.linalg.norm(vel_vector)
            if dist > 1e-8:    
                vel_vector += self.kr*(norm/dist)*(x[(robotNo + 1)%(N-1), :] - x[robotNo, :])
        
        if self.confirmed and self.mesured:
            if self.first:
                self.first = False
                zones_suspectes, valeurs = self.law_est.search_zones()
                print("Zones", zones_suspectes, valeurs)
                
                #self.pot.plot_essai(self.detected_sources[self.source_verif][1], self.detected_sources[self.source_verif][2])
                
                if (len(zones_suspectes) > 0):
                    i_max = 0
                    score_max = 0
                    for i in range(len(zones_suspectes)):
                        score = valeurs[i] + 200*(np.pi - 2*np.arctan(0.5*np.linalg.norm(self.detected_sources[self.source_verif][0] - zones_suspectes[i])))/(np.pi)
                        if score > score_max:
                            i_max = i
                
                    self.zones_recherche = (zones_suspectes[i_max], valeurs[i_max])
                    self.radius = 1
                    self.phase = 3
                    
                    print("Regroupement:", self.zones_recherche)
            
            else:
                ratio = (self.prev(x[robotNo, :]) / measurement[robotNo])
                if (ratio < 0.96)  or (ratio > 1.1):
                    self.zones_recherche = (x[robotNo, :], measurement[robotNo])
                    self.radius = 1
                    self.phase = 3
                    
                    print("Regroupement:", self.zones_recherche)
                
                elif self.radius > 15.:
                    self.cnt += 1
                
                    if self.cnt > 20:
                        self.radius = 1
                        self.phase = 3
                        
                        print("Regroupement:", self.zones_recherche)
        
        # initialize control input vector for current robot i
        dist_consensus = np.linalg.norm(vel_vector)
        if dist_consensus < 1e-8:
            factor_consensus = 0
        else:
            factor_consensus = self.kp_r*np.arctan(self.kpi_r*dist_consensus)/dist_consensus
        
        v = factor_consensus*vel_vector         

        # .................  TO BE COMPLETED HERE .............................
        return v
    
    # general template of a function defining a control law
    # =============================================================================
    def regroupement(self, t, robotNo, N, x, measurement):
    # ============================================================================= 
        
        if self.zones_recherche is None:
            v = np.zeros(2)
        else:
            if ((self.prev(x[robotNo, :]) / measurement[robotNo]) < 0.98) and (measurement[robotNo] > self.zones_recherche[1]): # Si on trouve un meilleur point durant le regroupement
                self.zones_recherche = (x[robotNo, :], measurement[robotNo])
                
            formation_distance = self.radius
            relative_pose = np.array([[formation_distance*np.sin(2*np.pi*i/N) for i in range(N)],       # x-coordinates (m)
                                        [formation_distance*np.cos(2*np.pi*i/N) for i in range(N)]]).T   # y-coordinates (m)
            
            sum_distances = 0.
            vel_vector = np.zeros(2)
            
            center_formation = np.array([np.mean(x[:, 0]), np.mean(x[:, 1])])
            for i in range(N):
                if i != robotNo:
                    dist = np.linalg.norm(x[robotNo, :] - x[i, :])
                    dist_rel = np.linalg.norm(relative_pose[robotNo, :] - relative_pose[i, :])
                    sum_distances += max((dist - dist_rel)/self.radius, 0)
                    vel_vector += ((dist - dist_rel)/dist)*(x[i, :] - x[robotNo, :])
            
            sum_distances += abs((dist - np.linalg.norm(relative_pose[robotNo, :]))/self.radius) # prendre en compte le fait d'arriver au niveau de l'objectif
            vel_vector += self.zones_recherche[0] - center_formation
            
            # initialize control input vector for current robot i
            dist_consensus = np.linalg.norm(vel_vector)
            if dist_consensus < 1e-8:
                factor_consensus = 0
            else:
                factor_consensus = self.kp_r*np.arctan(self.kpi_r*dist_consensus)/dist_consensus
            
            v = factor_consensus*vel_vector
            
            if (sum_distances < 1.5) and (np.linalg.norm(self.zones_recherche[0] - center_formation) < 0.5):
                self.phase = 1 
                
                print("Descente de gradient")         

        # .................  TO BE COMPLETED HERE .............................
        return v
# =============================================================================


# =============================================================================

def initialisation(N, difficulty=3, random=False, params = [2, 5, 0.1, 0.1, 0.1, 0.5], limits=[-25, 25, -25, 25, 0.5, 0.5], fenetre = 5):
    
    global pot
    global firstCall
    global controleur
    
    pot = Potential(difficulty=difficulty, random=random) 
    pot.get_truth()
    
    controleur = Controleur(N, pot, params, limits, fenetre)
    
    firstCall = False
    # --------------------------------


# =============================================================================
def potential_seeking_ctrl(t, robotNo, robots_poses):
# =============================================================================

    # --- example of modification of global variables ---
    # ---(updated values of global variables will be known at next call of this funtion) ---
    # global toto
    # toto = toto +1
    
    global firstCall
    global pot
    global controleur

    # --- part to be run only once --- 
    if firstCall:
    
        # !!!!!!!!!!!!!!!!!!!!!!!  DO NOT REMOVE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #     YOU CAN MODIFY difficulty {1,2,3} AND random {True, False} PARAMETERS
        pot = Potential(difficulty=3, random=False) 
        pot.get_truth() 
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        
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
    
    ui = controleur.commands(t, robotNo, N, x, pot_measurement)
    
    return ui[0], ui[1], pot   # potential is also returned to be used by main script for displays (DO NOT MODIFY)
# =============================================================================

def visu_solution():
    global controleur
    
    controleur.law_est.plot_fit(fit_actuel=False)

