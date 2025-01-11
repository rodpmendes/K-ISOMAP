"""

Unsupervised metric learning via K-ISOMAP for high-dimensional data clustering - Noise experiments
Created on Sat May 25 13:41:56 2024
Author: Gustavo H. Chavari

Code modified to incorporate functions in the experiments
Modified on Sat Jan 11 10:01:48 2025
Author: Rodrigo P.Mendes

Credits:
This script was created in conjunction with the article [Unsupervised metric learning via K-ISOMAP for high-dimensional data clustering]

"""
import openai
import logging
import warnings

# To avoid unnecessary warning messages
openai._utils._logs.logger.setLevel(logging.WARNING)
warnings.simplefilter(action='ignore')

import sys
import numpy as np
from numpy import sqrt
import repliclust
from sklearn.datasets import make_swiss_roll
from matplotlib.pyplot import cm
import matplotlib.pyplot as plt



def HopfLink(start=0, num=26, R=5, r=1, noise=True):
    """
    Generates a Hopf link by computing evenly spaced points in 3D space 
    for two interlinked circles (loops) on a torus.
    
    Parameters:
    - start (float): The starting value of the sequence.
    - num (int): The number of evenly spaced values to generate.
    - R (float): The major radius of the torus (distance from the torus center to the tube center).
    - r (float): The minor radius of the torus (radius of the tube itself).
    
    Returns:
    - result_matrix (np.ndarray): A matrix containing the 3D coordinates of the points
      from both loops, with an additional column indicating the loop (0 or 1).
    - nn (int): An integer derived from the total number of points (sqrt of the total count).

    Example:
    >>> result_matrix, nn = TorusLink(0, 100)
    >>> result_matrix.shape, nn
    ((200, 4), 14)
    """
    
    u = np.linspace(start, 2*np.pi, num)
    v = np.linspace(start, 2*np.pi, num)
    U, V = np.meshgrid(u, v)

    # Torus Surface equation
    x = (R+r*np.cos(V))*np.cos(U)
    y = (R+r*np.cos(V))*np.sin(U)-5
    z = r*np.sin(V)

    x_1 = r*np.sin(V) 
    y_1 = (R+r*np.cos(V))*np.sin(U) 
    z_1 = (R+r*np.cos(V))*np.cos(U)


    if noise:
        # Add Gaussian noise with 0.3 standard deviation
        np.random.seed(127)
        noise_matrix_1 = np.random.normal(0, 0.3, (len(z.flatten()), 3))
        noise_matrix_2 = np.random.normal(0, 0.3, (len(z.flatten()), 3))

        data_matrix_1 = np.column_stack((x.flatten(), y.flatten(), z.flatten())) + noise_matrix_1
        data_matrix_2 = np.column_stack((x_1.flatten(), y_1.flatten(), z_1.flatten())) + noise_matrix_2
        
        result_matrix = np.vstack([
                            np.column_stack((data_matrix_1, np.full(len(z.flatten()), 0))),
                            np.column_stack((data_matrix_2, np.full(len(z_1.flatten()), 1)))
                        ]).astype('float64')
    else:
        data_matrix_1 = np.column_stack((x.flatten(), y.flatten(), z.flatten()))
        data_matrix_2 = np.column_stack((x_1.flatten(), y_1.flatten(), z_1.flatten()))

        result_matrix = np.vstack([
                            np.column_stack((data_matrix_1, np.full(len(z.flatten()), 0))),
                            np.column_stack((data_matrix_2, np.full(len(z_1.flatten()), 1)))
                        ]).astype('float64')

    
    # Plot points
    fig = plt.figure(figsize=(6,6))
    # Hopf link
    ax1 = fig.add_subplot(111, projection='3d')  
    ax1.scatter(result_matrix[:,:3].T[0], result_matrix[:,:3].T[1], result_matrix[:,:3].T[2], c=[cm.rainbow(valor) for valor in result_matrix.T[3]], alpha=0.5)
    ax1.view_init(elev=30, azim=30)
    ax1.set_title('Hopf Link',pad=16)  
    ax1.axis('off')
    ax1.set_xlim3d(-10, 5)
    ax1.set_ylim3d(-10, 5)
    ax1.set_zlim3d(-7, 5)
    ax1.set_xlim(-10, 5)
    ax1.set_ylim(-10, 5)
    # Shadow = Projecting the points onto the xy-plane by plotting them with a fixed z-coordinate that matches the lower z-limit.
    ax1.scatter(result_matrix.T[0], result_matrix.T[1], -7*np.ones_like(result_matrix.T[2]), c='gray',alpha=0.02)

    #plt.savefig('hopf_link.jpeg',format='jpeg',dpi=300)
    #plt.close()
    
    return result_matrix[:,:3] , result_matrix.T[3]

def RepliclustArchetype():

    repliclust.base.set_seed(7)

    archetype = repliclust.Archetype(
    min_overlap=0.01, max_overlap=0.05,
    dim=3,
    n_samples=1100,
    distributions=[('gamma', {'shape': 1, 'scale': 2.0})]
    )

    X1, y1, _ = (repliclust.DataGenerator(archetype).synthesize(quiet=True))

    X1 = X1.astype(np.float64)


    # Add noise
    magnitude = np.linspace(0,1,26)
    ruido = np.random.normal(0, scale=magnitude[0], size=X1.shape)

    # Surface equation
    X1.T[0] = X1.T[0] + ruido.T[0]
    X1.T[1] = X1.T[1] + ruido.T[1]
    X1.T[2] = X1.T[2] + ruido.T[2]

     # Hopf link
    fig = plt.figure(figsize=(6,6))
    ax1 = fig.add_subplot(111, projection='3d')  
    ax1.scatter(X1.T[0],X1.T[1], X1.T[2], c=y1, alpha=0.5)
    ax1.view_init(elev=30, azim=30)
    ax1.set_title('Repliclust Archetype Gamma')  
    ax1.axis('off')
    ax1.set_xlim3d(-10, 5)
    ax1.set_ylim3d(-10, 5)
    ax1.set_zlim3d(-7, 5)
    ax1.set_xlim(-10, 5)
    ax1.set_ylim(-10, 5)
    # Shadow = Projecting the points onto the xy-plane by plotting them with a fixed z-coordinate that matches the lower z-limit.
    ax1.scatter(X1.T[0], X1.T[1], -7*np.ones_like(X1.T[2]), c='gray',alpha=0.02)

    #plt.savefig('repliclust_archetype_gamma.jpeg',format='jpeg',dpi=300)
    #plt.close()

    return X1, y1

def SwissRoll():

    # Parametrization
    X2, color = make_swiss_roll(n_samples=2000,hole=False)

    X2 = np.array(X2)

    # SwissRoll
    fig = plt.figure(figsize=(6,6))
    ax1 = fig.add_subplot(111, projection='3d')  
    ax1.scatter(X2.T[0],X2.T[1], X2.T[2], c=color, alpha=0.5)
    ax1.set_title('Swiss Roll')  
    ax1.axis('off')
    # Shadow = Projecting the points onto the xy-plane by plotting them with a fixed z-coordinate that matches the lower z-limit.
    ax1.scatter(X2.T[0], X2.T[1], -7*np.ones_like(X2.T[2]), c='gray',alpha=0.02)

    #plt.savefig('swiss_roll.jpeg',format='jpeg',dpi=300)
    #plt.close()

    return X2, color

def main():
    RepliclustArchetype()
    SwissRoll()
    HopfLink(noise=True)
    
if __name__ == '__main__':
    sys.exit(main())