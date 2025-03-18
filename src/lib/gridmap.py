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
    def __init__(self, xmin, xmax, ymin, ymax, dx, dy):
    # -------------------------------------------------------------------------

        self.xmin = xmin
        self.xmax = xmax
        self.dx = dx
        self.ymin = ymin
        self.ymax = ymax
        self.dy = dy

        # map of intensities values from t = t0 to t = tf
        self.map = np.zeros((int(round((self.ymax-self.ymin)/self.dy, 0)), int(round((self.xmax-self.xmin)/self.dx, 0))))
        print(np.shape(self.map))

        # number of times a cell of the map had its value updated
        self.map_occurences = np.zeros((int(round((self.ymax-self.ymin)/self.dy, 0)), int(round((self.xmax-self.xmin)/self.dx, 0))))
        print(np.shape(self.map_occurences))
        
    def update(self, xpos, ypos, particle_intensity_value, t):
        
        #print(ypos, self.ymin, self.dy, self.ymax)
        #print(-(ypos - self.ymin)/self.dy + (self.ymax-self.ymin)/self.dy - self.dy)
        #print(round(-(ypos - self.ymin)/self.dy + (self.ymax-self.ymin)/self.dy - self.dy, 0))
        #print(int(round(-(ypos - self.ymin)/self.dy + (self.ymax-self.ymin)/self.dy - self.dy, 0)))
        map_line_index = int(round(-(ypos - self.ymin)/self.dy + (self.ymax-self.ymin)/self.dy - self.dy, 0))
        map_column_index = int(round((xpos - self.xmin)/self.dx - self.dx, 0))
        
        # incrémenter nombre de fois que la cellule a été update
        self.map_occurences[map_line_index, map_column_index] += 1
        
        occurence = self.map_occurences[map_line_index, map_column_index]
        current_partiucle_intensity_value = self.map[map_line_index, map_column_index]
        
        # update map particle value pondérée
        self.map[map_line_index, map_column_index] = (current_partiucle_intensity_value*(occurence-1) + particle_intensity_value)/occurence

            
        

        print(f"updating position {xpos}x{ypos} at time {t}")
        print(f"total position updates : {self.map_occurences[map_line_index, map_column_index]}")
        print(f"position in table : line = {map_line_index}, column = {map_column_index}")

        


    def plot(self, t=0., pause_time = 0.01, permanent = False):

        self.fig = plt.figure("Potential map")
        self.ax = self.fig.add_subplot(111)
        self.ax.imshow(self.map[:, :])
            # ax.set_xlim(self.xmin, self.xmax)
            # ax.set_ylim(self.ymax, self.ymin)
        self.ax.set_title(f"Map potential discovery over time : t = {t}")
        if pause_time != 0:
            plt.pause(pause_time)
            
            
        self.ax.cla()
        self.ax.imshow(self.map[:, :])
        self.ax.set_title(f"Map potential discovery over time : t = {t}")
        if pause_time != 0:
            plt.pause(pause_time)
        
        if permanent:
            plt.show()
            
        
        print("map plotted")

    # def plot_animation(self, tf = None):        
        
    #     if tf == None:
    #         tf = self.tf

    #     fig = plt.figure("Potential map over time")
    #     ax = fig.add_subplot(111, aspect='equal', autoscale_on=False, xlim=(self.xmin, self.xmax), ylim=(self.ymin, self.ymax))
        
    #     for t in range(self.t0, tf, self.dt):

    #         ax.cla()
    #         ax.set_title(f"Map potential discovery over time : t = {t}/{self.tf}")

    #         self.plot(fig, ax, t)


        








# ============================== MAIN =========================================        
if __name__ =='__main__':
# =============================================================================

    xmin = -10
    xmax = 10
    
    ymin = -10
    ymax = 10

    dx = 0.1
    dy = 0.2
    
    
    grid_test = gridmap(xmin, xmax, ymin, ymax, dx, dy)
    grid_test.update(-9, -9, 15, 1)
    fig, ax = grid_test.plot(1, pause_time=1)
    
    grid_test.update(0, 0, 15, 1)
    fig, ax = grid_test.plot(2, fig, ax,  pause_time=1)
    
    grid_test.update(0, 0, 45, 1)
    fig, ax = grid_test.plot(3, fig, ax,  pause_time=1)

    plt.show()

        








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

