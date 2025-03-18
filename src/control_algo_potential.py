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
from lib.potential import Potential
from lib.gridmap import gridmap


# ==============   "GLOBAL" VARIABLES KNOWN BY ALL THE FUNCTIONS ==============
# all variables declared here will be known by functions below
# use keyword "global" inside a function if the variable needs to be modified by the function

# global toto

global firstCall   # can be used to check the first call ever of a function
firstCall = True

global pot # DO NOT MODIFY - allows initialisation of potential function from this script
global detected_sources
detected_sources = {}

global gridmap_record

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
    


    # --- part to be run only once --- 
    if firstCall:
    
        # !!!!!!!!!!!!!!!!!!!!!!!  DO NOT REMOVE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #     YOU CAN MODIFY difficulty {1,2,3} AND random {True, False} PARAMETERS
        pot = Potential(difficulty=3, random=True)  
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        gridmap_record = gridmap(-25, 25, -25, 25, 0.5, 0.5) # A bit dangerous to do so because not 
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
        
        print(x[m, :])
        gridmap_record.update(x[m, 0], x[m, 1], pot_measurement[m], t)

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # to get access to potential measurement from robot i at time t in the rest of the code
    # you can use eihter use    pot_measurement[i]     or      pot.value(x[i,:])
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
    ui = formation_gradient(t, robotNo, N, x, pot_measurement)
    
    
    return ui[0], ui[1], pot   # potential is also returned to be used by main script for displays (DO NOT MODIFY)
# =============================================================================

def get_gridmap():
    global gridmap_record
    
    return gridmap_record



# general template of a function defining a control law
# =============================================================================
def formation_gradient(t, robotNo, N, x, measurement):
# ============================================================================= 
    
    global detected_sources

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
            sum_distances += max(dist - formation_distance, 0)
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
    kp = 3
    kg = 5
    kto = 0.1
    dist_consensus = np.linalg.norm(vel_vector)
    factor_consensus = kp*np.arctan(dist_consensus)/dist_consensus
    
    dist_grad = np.linalg.norm(grad_total)
    if dist_grad < 0.0001:
        factor_grad = 0
    else:
        factor_grad = (kg*np.arctan(dist_grad)/dist_grad)*(np.pi - 2*np.arctan(sum_distances*kto))/(np.pi)
    
    v = factor_consensus*vel_vector - factor_grad*grad_total
    
    if (sum_distances < 0.2) and (dist_grad < 0.1):
        potential_source = np.array([np.mean(x[:, 0]), np.mean(x[:, 1])])
        for source in detected_sources:
            if np.linalg.norm(source -potential_source) < 0.1:
                break
        
        else:
            detected_sources[potential_source] = ([[1.0, 0.], [0., 1.]], 10)

    # .................  TO BE COMPLETED HERE .............................
    return v
# =============================================================================

