# -*- coding: utf-8 -*-
"""
Potential class

(c) S. Bertrand, 2024
"""


import numpy as np

import matplotlib.pyplot as plt

from scipy.stats import multivariate_normal



# =============================================================================
class Potential:
# =============================================================================
    
    # -------------------------------------------------------------------------
    def __init__(self, difficulty=1, random=False):
    # -------------------------------------------------------------------------    
        if (difficulty<1)or(difficulty>3):
            raise NameError("Difficulty must be >=1 and <=3")
        
        self.difficulty = difficulty
        self.random = random
        
        self.xmin = -25.
        self.xmax = 25.
        self.xstep = 0.05
        self.ymin = -25.
        self.ymax = 25.
        self.ystep = 0.05
        
        
        
        if (random):
            xwidth = np.abs(self.xmax - self.xmin)
            ywidth = np.abs(self.ymax - self.ymin)
            self.mu1 = [ 0.6*(xwidth*np.random.rand()-xwidth/2.) , 0.6*(ywidth*np.random.rand()-ywidth/2.) ]
            self.mu2 = [ xwidth*np.random.rand()-xwidth/2. , ywidth*np.random.rand()-ywidth/2. ]
            self.mu3 = [ xwidth*np.random.rand()-xwidth/2. , ywidth*np.random.rand()-ywidth/2. ]
        else:
            self.mu1 = [6, 4]
            self.mu2 = [-2, -2]
            self.mu3 = [-7, 10]
        
        self.gaussian1 = multivariate_normal(self.mu1, [[1.0, 0.], [0., 1.]])
        self.gaussian2 = multivariate_normal(self.mu2, [[0.5, 0.3], [0.3, 0.5]])
        self.gaussian3 = multivariate_normal(self.mu3, [[0.8, 0.], [0., 0.8]])
        
        self.weight1 = 10000
        self.weight2 = 1
        self.weight3 = 1E-8
        
        
        self.mu = [self.mu1]
        if (difficulty>1):
            self.mu.append(self.mu2)
            if (difficulty>2):
                self.mu.append(self.mu3)
        
        self.distribution = [self.gaussian1]
        if (difficulty>1):
            self.distribution.append(self.gaussian2)
            if (difficulty>2):
                self.distribution.append(self.gaussian3)
                
        self.weight = [self.weight1]
        if (difficulty>1):
            self.weight.append(self.weight2)
            if (difficulty>2):
                self.weight.append(self.weight3)


    
    # -------------------------------------------------------------------------
    def value(self,pos):  # (pos = [x,y]) 
    # -------------------------------------------------------------------------
        sumval = 0.
        
        for i in range(self.difficulty):
            sumval += self.weight[i]*self.distribution[i].pdf(pos)
        
        return 310. + np.log10(sumval + 1e-309)


    # -------------------------------------------------------------------------
    def plot(self,noFigure=None,fig=None,ax=None, colorbar=True):
    # -------------------------------------------------------------------------
        x, y = np.mgrid[self.xmin:self.xmax:self.xstep, self.ymin:self.ymax:self.ystep]
        pos = np.dstack((x, y))
        potentialFieldForPlot = self.value(pos)
        
        if (fig==None):
            if (noFigure==None):
                noFigure=1
            fig = plt.figure(noFigure)
        if (ax==None):
            ax = fig.add_subplot(111)
            
        # Séparation des coordonnées x et y
        x_values, y_values = zip(*self.mu)
        cs = ax.contourf(x, y, potentialFieldForPlot, 20, cmap='BrBG')
        ax.scatter(x_values, y_values)
        #cs = ax.contour(x, y, potentialFieldForPlot, 10, cmap='BrBG')
        
        if (colorbar):
            fig.colorbar(cs)
        
        return fig, ax
    
    
    def plot_essai(self, weight, distr, colorbar=True):
        x, y = np.mgrid[self.xmin:self.xmax:self.xstep, self.ymin:self.ymax:self.ystep]
        pos = np.dstack((x, y))
        potentialFieldForPlot = self.value(pos)
        
        potential = 310. + np.log10(weight*distr.pdf(pos) + 1e-309)
        
        ratio1 = np.fmax(potential / potentialFieldForPlot, 0.)
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cs = ax.contourf(x, y, potentialFieldForPlot, 20, cmap='BrBG')
        
        if (colorbar):
            fig.colorbar(cs)
        
        x_values, y_values = zip(*self.mu)
        fig, axes = plt.subplots(1, 1, figsize=(15, 5))

        cs1 = axes.contourf(x, y, ratio1, 20, cmap='coolwarm', vmin=0, vmax=1)
        axes.set_title("Ratio: PotentialFieldForPlot / Potential 1")
        axes.scatter(x_values[0], y_values[0])
        plt.colorbar(cs1, ax=axes)
        
        plt.tight_layout()
        
        # Création d'une image RGB vide (initialement noire)
        color_map = np.zeros((*ratio1.shape, 3))  # (height, width, 3) pour R, G, B

        # Appliquer les couleurs en fonction des ratios
        color_map[..., 0] = (ratio1 > 0.95).astype(float).T  # Rouge si ratio1 > 0.95

        # Affichage de l'image colorée
        plt.figure(figsize=(6,6))
        plt.imshow(color_map, extent=[x.min(), x.max(), y.min(), y.max()], origin="lower")
        plt.scatter(x_values, y_values)
        plt.title("Zones où ratio > 0.95")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()
    
    # -------------------------- -----------------------------------------------
    def plot_test(self, noFigure=None,fig=None,ax=None, colorbar=True):
    # -------------------------------------------------------------------------
        x, y = np.mgrid[self.xmin:self.xmax:self.xstep, self.ymin:self.ymax:self.ystep]
        pos = np.dstack((x, y))
        potentialFieldForPlot = self.value(pos)
        
        potential1 = 310. + np.log10(self.weight[0]*self.distribution[0].pdf(pos) + 1e-309)
        potential2 = 310. + np.log10(self.weight[1]*self.distribution[1].pdf(pos) + 1e-309)
        potential3 = 310. + np.log10(self.weight[1]*self.distribution[2].pdf(pos) + 1e-309)
        
        ratio1 = np.fmax(potential1 / potentialFieldForPlot, 0.)
        ratio2 = np.fmax(potential2 / potentialFieldForPlot, 0.)
        ratio3 = np.fmax(potential3 / potentialFieldForPlot, 0.)
        
        if (fig==None):
            if (noFigure==None):
                noFigure=1
            fig = plt.figure(noFigure)
        if (ax==None):
            ax = fig.add_subplot(111)
        cs = ax.contourf(x, y, potentialFieldForPlot, 20, cmap='BrBG')
        
        if (colorbar):
            fig.colorbar(cs)
        
        x_values, y_values = zip(*self.mu)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        cs1 = axes[0].contourf(x, y, ratio1, 20, cmap='coolwarm', vmin=0, vmax=1)
        axes[0].set_title("Ratio: PotentialFieldForPlot / Potential 1")
        axes[0].scatter(x_values[0], y_values[0])
        plt.colorbar(cs1, ax=axes[0])

        cs2 = axes[1].contourf(x, y, ratio2, 20, cmap='coolwarm', vmin=0, vmax=1)
        axes[1].set_title("Ratio: PotentialFieldForPlot / Potential 2")
        axes[1].scatter(x_values[1], y_values[1])
        plt.colorbar(cs2, ax=axes[1])

        cs3 = axes[2].contourf(x, y, ratio3, 20, cmap='coolwarm', vmin=0, vmax=1)
        axes[2].set_title("Ratio: PotentialFieldForPlot / Potential 3")
        axes[2].scatter(x_values[2], y_values[2])
        plt.colorbar(cs3, ax=axes[2])
        
        plt.tight_layout()
        
        # Création d'une image RGB vide (initialement noire)
        color_map = np.zeros((*ratio1.shape, 3))  # (height, width, 3) pour R, G, B

        # Appliquer les couleurs en fonction des ratios
        color_map[..., 0] = (ratio1 > 0.99).astype(float).T  # Rouge si ratio1 > 0.95
        color_map[..., 1] = (ratio2 > 0.99).astype(float).T  # Vert si ratio2 > 0.95
        color_map[..., 2] = (ratio3 > 0.99).astype(float).T  # Bleu si ratio3 > 0.95

        # Affichage de l'image colorée
        plt.figure(figsize=(6,6))
        plt.imshow(color_map, extent=[x.min(), x.max(), y.min(), y.max()], origin="lower")
        plt.scatter(x_values, y_values)
        plt.title("Zones où ratio > 0.95")
        plt.xlabel("X")
        plt.ylabel("Y")
        
        return fig, ax

    # -------------------------------------------------------------------------
    def grad(self, pos1, pos2):
    # -------------------------------------------------------------------------
        g = (self.value(pos2) - self.value(pos1)) / np.linalg.norm(pos2-pos1)
        grad =  np.array([g*(pos1[0] - pos2[0]), g*(pos1[1] - pos2[1])])
        return grad
    
    
    # -------------------------------------------------------------------------    
    def meanGrad(self,point, epsilon=1.0):
    # -------------------------------------------------------------------------
        p1 = point.copy()
        p1n = []
        # neighborhood
        p1n.append(p1 + np.array([-epsilon,epsilon]))
        p1n.append(p1 + np.array([epsilon,epsilon]))
        p1n.append(p1 + np.array([epsilon,-epsilon]))
        p1n.append(p1 + np.array([-epsilon,-epsilon]))
        
        
        p1nGrad = []
        meanGrad = np.array([0,0])
        for pt in p1n:
            ptGrad = self.grad(p1, pt)
            
            meanGrad[0] += ptGrad[0]
            meanGrad[1] += ptGrad[1]
            
            p1nGrad.append(ptGrad)
        

        meanGrad = -meanGrad/4.
        
        return meanGrad


    # -------------------------------------------------------------------------    
    def plotQuiverMeanGrad(self, point, epsilon, ax):
    # -------------------------------------------------------------------------
        p1 = point.copy()
        p1n = []
        # neighborhood
        p1n.append(p1 + np.array([-epsilon,epsilon]))
        p1n.append(p1 + np.array([epsilon,epsilon]))
        p1n.append(p1 + np.array([epsilon,-epsilon]))
        p1n.append(p1 + np.array([-epsilon,-epsilon]))
        
        
        p1nGrad = []
        meanGrad = np.array([0,0])
        for pt in p1n:
            ptGrad = self.grad(p1, pt)
            
            #print(ptGrad)
            
            meanGrad[0] += ptGrad[0]
            meanGrad[1] += ptGrad[1]
            
            p1nGrad.append(ptGrad)
            #plt.quiver(pt[0],pt[1],ptGrad[0],ptGrad[1])
        
        #print(meanGrad)
        meanGrad = -meanGrad/4.
        ax.quiver(pt[0],pt[1],meanGrad[0],meanGrad[1])
        
        ax.text(pt[0]+epsilon,pt[1],  '%.2f' % (180./np.pi*np.arctan2(meanGrad[1],meanGrad[0])) )
    
        return meanGrad
    
    def get_truth(self):
        print(self.mu)


# ======================== END OF CLASS Potential =============================

def g(x, mu, Sigma, alpha=1.0):
    """
    Fonction qui a les mêmes lignes de niveau que le logarithme d'une gaussienne,
    mais décroît moins vite et se stabilise à 0.
    
    Paramètres :
        x : np.array, vecteur de point où évaluer la fonction
        mu : np.array, vecteur moyen de la gaussienne
        Sigma : np.array, matrice de covariance (doit être inversible)
        alpha : float, paramètre de contrôle de la décroissance
    
    Retour :
        float, valeur de g(x)
    """
    diff = x - mu
    Sigma_inv = np.linalg.inv(Sigma)
    quad_form = np.dot(diff.T, np.dot(Sigma_inv, diff))
    return -1 / (1 + alpha * quad_form)

# Exemple d'utilisation
mu = np.array([0, 0])  # Moyenne en 2D
Sigma = np.array([[1, 0], [0, 1]])  # Matrice de covariance identitaire
x = np.array([1, 1])  # Point où évaluer g

g_value = g(x, mu, Sigma, alpha=0.5)
print("g(x) =", g_value)




# ========================== MAIN =============================================
if __name__=='__main__':
# =============================================================================
    
    plt.close()
    
    pot = Potential(difficulty=3, random=True)
    
    
    fig2, ax2 = pot.plot(2)
    

    print(pot.value(pot.mu1))
    print(pot.value(pot.mu2))
    print(pot.value(pot.mu3))
    
    
    
    epsilon = 1.0
    
    
    for xx in np.arange(-20,25,5):
        for yy in np.arange(-20,25,5):
            point = np.array([xx,yy])
            pot.plotQuiverMeanGrad(point, epsilon,ax2)
    
    v=pot.meanGrad(np.array([10,10]), epsilon)
    print(v)
    print(np.arctan2(v[1],v[0])*180./np.pi)    
    
    
    
    pot.plot(1)
    pot.plot_test(3)
    
    plt.show()
