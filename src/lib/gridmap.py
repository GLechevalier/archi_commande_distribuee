# -*- coding: utf-8 -*-
"""
gridmap class

Arnaud HUILLET, 2025
"""


import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
class gridmap:
# =============================================================================
    
    # -------------------------------------------------------------------------
    def __init__(self, xmin, xmax, ymin, ymax, dx, dy, t0, tf, dt):
    # -------------------------------------------------------------------------

        self.xmin = xmin
        self.xmax = xmax
        self.dx = dx
        self.ymin = ymin
        self.ymax = ymax
        self.dy = dy
        self.t0 = t0
        self.tf = tf
        self.dt = dt

        # map of intensities values from t = t0 to t = tf
        self.map = -np.ones((int((ymax-ymin)/dy), int((xmax-xmin)/dx), int(round((tf-t0+dt)/dt, 0))))

        print(np.shape(self.map))

        # number of times a cell of the map had its value updated
        self.map_occurences = np.zeros((int((ymax-ymin)/dy), int((xmax-xmin)/dx)))
    
    def update(self, xpos, ypos, particle_intensity_value, t):

        map_line_index = int(-(ypos-ymin)/dy + (ymax-ymin)/dy - dy)
        map_column_index = int((xpos-xmin)/dx - dx)
        self.map[map_line_index, map_column_index, int(t/dt)] = particle_intensity_value
        self.map_occurences[map_line_index, map_column_index] += 1

        print(f"updating position {xpos}x{ypos} at time {t}")
        print(f"position in table : line = {map_line_index}, column = {map_column_index}")

        


    def plot(self, t, fig = None, ax = None):

        if ax == None or fig == None:
            fig = plt.figure("Potential map over time")
            ax = fig.add_subplot(111)
            plt.imshow(self.map[:, :, int(t/self.dt)])
            ax.set_xlim = (self.xmin, self.xmax)
            ax.set_ylim = (self.ymax, self.ymin)
            ax.set_title(f"Map potential discovery over time : t = {t}")
            
            
            
        else:
            ax.imshow(self.map[:, :, int(t/self.dt)])
        
        print("map plotted")


    def plot_animation(self, tf = None):        
        
        if tf == None:
            tf = self.tf

        fig = plt.figure("Potential map over time")
        ax = fig.add_subplot(111, aspect='equal', autoscale_on=False, xlim=(self.xmin, self.xmax), ylim=(self.ymin, self.ymax))
        
        for t in range(self.t0, tf, self.dt):

            ax.cla()
            ax.set_title(f"Map potential discovery over time : t = {t}/{self.tf}")

            self.plot(fig, ax, t)


        








# ============================== MAIN =========================================        
if __name__=='__main__':
# =============================================================================

    xmin = -10
    xmax = 10
    
    ymin = -10
    ymax = 10

    dx = 0.1
    dy = 0.2

    t0 = 0
    tf = 5
    dt = 0.1

    grid_test = gridmap(xmin, xmax, ymin, ymax, dx, dy, t0, tf, dt)
    grid_test.update(0, 0, 15, 1)
    grid_test.plot(1)

    plt.show()

