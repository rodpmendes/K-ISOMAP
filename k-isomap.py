#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Unsupervised metric learning via K-ISOMAP for high-dimensional data clustering

Created on Wed Jul 24 16:59:33 2023

"""

# Imports
import sys
import time
import warnings
import numpy as np
import scipy as sp
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import sklearn.datasets as skdata
import sklearn.neighbors as sknn
import sklearn.utils.graph as sksp
from numpy import log
from numpy import trace
from numpy import dot
from numpy import sqrt
from numpy.linalg import det
from numpy.linalg import inv
from numpy.linalg import norm
from scipy.spatial.distance import cosine
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import Isomap
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import train_test_split
from sklearn.metrics.cluster import rand_score
from sklearn.metrics.cluster import silhouette_score
from sklearn.metrics.cluster import mutual_info_score
from sklearn.metrics.cluster import homogeneity_score
from sklearn.metrics.cluster import completeness_score
from sklearn.metrics.cluster import v_measure_score
from sklearn.metrics.cluster import fowlkes_mallows_score
from sklearn.metrics.cluster import calinski_harabasz_score
from sklearn.metrics.cluster import davies_bouldin_score

# To avoid unnecessary warning messages
warnings.simplefilter(action='ignore')

#################################################################################
# Fast K-ISOMAP implementation - consumes more memory, but it is a lot faster
# Pre-allocate matrices to speed up performance
#################################################################################
def KIsomap(dados, k, d, option, alpha=0.5):
    # Number of samples and features  
    n = dados.shape[0]
    m = dados.shape[1]
    # Matrix to store the principal components for each neighborhood
    matriz_pcs = np.zeros((n, m, m))
    # Generate KNN graph
    knnGraph = sknn.kneighbors_graph(dados, n_neighbors=k, mode='distance')
    A = knnGraph.toarray()
    # Computes the means and covariance matrices for each patch
    for i in range(n):       
        vizinhos = A[i, :]
        indices = vizinhos.nonzero()[0]
        if len(indices) == 0:                   # Treat isolated points
            matriz_pcs[i, :, :] = np.eye(m)     # Eigenvectors in columns
        else:
            # Get the neighboring samples
            amostras = dados[indices]
            v, w = np.linalg.eig(np.cov(amostras.T))
            # Sort the eigenvalues
            ordem = v.argsort()
            # Select the d eigenvectors associated to the d largest eigenvalues
            maiores_autovetores = w[:, ordem[::-1]]                 
            # Projection matrix
            Wpca = maiores_autovetores  # Autovetores nas colunas
            #print(Wpca.shape)
            matriz_pcs[i, :, :] = Wpca
        
    # Defines the patch-based matrix (graph)
    B = A.copy()
    for i in range(n):
        for j in range(n):
            if B[i, j] > 0:
                delta = norm(matriz_pcs[i, :, :] - matriz_pcs[j, :, :], axis=0)
                
                ##### Functions of the principal curvatures (definition of the metric)
                # We must choose one single option for each execution
                if option == 0:
                    B[i, j] = norm(delta)                  # metric A0 - Norms of the principal curvatures
                elif option == 1:
                    B[i, j] = delta[0]                      # metric A1 - Curvature of the first principal component
                elif option == 2:
                    B[i, j] = delta[-1]                    # metric A2 - Curvature of the last principal component
                elif option == 3:
                    B[i, j] = (delta[0] + delta[-1])/2     # metric A3 - Average between the curvatures of first and last principal components
                elif option == 4:
                    B[i, j] = np.sum(delta)/len(delta)     # metric A4 - Mean curvature
                elif option == 5:
                    B[i, j] = max(delta)                   # metric A5 - Maximum curvature
                elif option == 6:
                    B[i, j] = min(delta)                   # metric A6 - Minimum curvature
                elif option == 7:
                    B[i, j] = min(delta)*max(delta)        # metric A7 - Product between minimum and maximum curvatures
                elif option == 8:
                    B[i, j] = max(delta) - min(delta)      # metric A8 - Difference between maximum and minimum curvatures
                elif option == 9:
                    B[i, j] = 1 - np.exp(-delta.mean())     # metric A9 - Negative exponential kernel
                else:
                    B[i, j] = ((1-alpha)*A[i, j]/sum(A[i, :]) + alpha*norm(delta))      # alpha = 0 => regular ISOMAP, alpha = 1 => K-ISOMAP 
                
    # Computes geodesic distances using the previous selected metric
    G = nx.from_numpy_array(B)
    D = nx.floyd_warshall_numpy(G)  
    # Computes centering matrix H
    H = np.eye(n, n) - (1/n)*np.ones((n, n))
    # Computes the inner products matrix B
    B = -0.5*H.dot(D**2).dot(H)    
    # Remove infs and nans from B (if the graph is not connected)
    maximo = np.nanmax(B[B != np.inf])   
    B[np.isnan(B)] = 0
    B[np.isinf(B)] = maximo
    # Eigeendecomposition
    lambdas, alphas = np.linalg.eig(B)
    # Sort eigenvalues and eigenvectors
    indices = lambdas.argsort()[::-1]
    lambdas = lambdas[indices]
    alphas = alphas[:, indices]
    # Select the d largest eigenvectors
    lambdas = lambdas[0:d]
    alphas = alphas[:, 0:d]
    # Computes the intrinsic coordinates
    output = alphas*np.sqrt(lambdas)    
    # Return the low dimensional coordinates
    return output.real


'''
 Performs clustering in data, returns the obtained labels and evaluates the clusters
'''
def Clustering(dados, target, DR_method, cluster):
    print()
    print('Clustering results for %s features' %(DR_method))
    print()
    # Number of classes
    c = len(np.unique(target))
    # Clustering algorithm
    if cluster == 'kmeans':
        kmeans = KMeans(n_clusters=c, random_state=42).fit(dados.T)
        labels = kmeans.labels_
    elif cluster == 'gmm':
        labels = GaussianMixture(n_components=c, random_state=42).fit_predict(dados.T)
    else:
        ward = AgglomerativeClustering(n_clusters=c, linkage='ward').fit(dados.T)
        labels = ward.labels_
    # Computation of the cluster evaluation metrics    
    rand = rand_score(target, labels)    
    ca = calinski_harabasz_score(dados.T, labels)
    fm = fowlkes_mallows_score(target, labels)
    v = v_measure_score(target, labels)
    # Print evaluation metrics
    print('Rand index: ', rand)    
    print('Calinski Harabasz: ', ca)
    print('Fowlkes Mallows:', fm)
    print('V measure:', v)
    print()

    return [rand, ca, fm, v, labels]



'''
Produces scatter plots of the 2D mappings
'''
def PlotaDados(dados, labels, metodo):
    # Number of classes
    nclass = len(np.unique(labels))
    # Converts list to an array
    rotulos = np.array(labels)
    # Define colors according to the number of classes
    if nclass > 11:
        #cores = ['black', 'gray', 'silver', 'whitesmoke', 'rosybrown', 'firebrick', 'red', 'darksalmon', 'sienna', 'sandybrown', 'bisque', 'tan', 'moccasin', 'floralwhite', 'gold', 'darkkhaki', 'lightgoldenrodyellow', 'olivedrab', 'chartreuse', 'palegreen', 'darkgreen', 'seagreen', 'mediumspringgreen', 'lightseagreen', 'paleturquoise', 'darkcyan', 'darkturquoise', 'deepskyblue', 'aliceblue', 'slategray', 'royalblue', 'navy', 'blue', 'mediumpurple', 'darkorchid', 'plum', 'm', 'mediumvioletred', 'palevioletred', ]
        cores = list(mcolors.CSS4_COLORS.keys())
        np.random.shuffle(cores)
    else:
        cores = ['blue', 'red', 'cyan', 'black', 'orange', 'magenta', 'green', 'darkkhaki', 'brown', 'purple', 'salmon']
    # Create figure
    plt.figure(10)
    for i in range(nclass):
        indices = np.where(rotulos==i)[0]        
        cor = cores[i]
        plt.scatter(dados[indices, 0], dados[indices, 1], c=cor, marker='*')
    # Save figure in image fila
    nome_arquivo = metodo + '.png'
    plt.title(metodo+' clusters')
    plt.savefig(nome_arquivo)
    plt.close()


##################### Beginning of thescript

#####################  Data loading

# First set of experiments
X = skdata.fetch_openml(name='servo', version=1)                      
#X = skdata.fetch_openml(name='car-evaluation', version=1)            
#X = skdata.fetch_openml(name='breast-tissue', version=2)
#X = skdata.fetch_openml(name='Engine1', version=1)                       
#X = skdata.fetch_openml(name='xd6', version=1)                         
#X = skdata.fetch_openml(name='heart-h', version=3)
#X = skdata.fetch_openml(name='steel-plates-fault', version=3)
#X = skdata.fetch_openml(name='PhishingWebsites', version=1)                # 10% of the samples
#X = skdata.fetch_openml(name='satimage', version=1)                        # 25% of the samples
#X = skdata.fetch_openml(name='led24', version=1)                           # 20% of the samples
#X = skdata.fetch_openml(name='hayes-roth', version=2)
#X = skdata.fetch_openml(name='rabe_131', version=2)
#X = skdata.fetch_openml(name='prnn_synth', version=1)                  
#X = skdata.fetch_openml(name='visualizing_environmental', version=2)   
#X = skdata.fetch_openml(name='diggle_table_a2', version=2)             
#X = skdata.fetch_openml(name='newton_hema', version=2)                 
#X = skdata.fetch_openml(name='wisconsin', version=2)                   
#X = skdata.fetch_openml(name='fri_c4_250_100', version=2)              
#X = skdata.fetch_openml(name='conference_attendance', version=1)       
#X = skdata.fetch_openml(name='tic-tac-toe', version=1)     
#X = skdata.fetch_openml(name='qsar-biodeg', version=1)     
#X = skdata.fetch_openml(name='spambase', version=1)                        # 25% of the samples
#X = skdata.fetch_openml(name='cmc', version=1)     
#X = skdata.fetch_openml(name='heart-statlog', version=1)

# Second set of experiments
#X = skdata.fetch_openml(name='cnae-9', version=1)                          # 50-D 
#X = skdata.fetch_openml(name='AP_Breast_Kidney', version=1)                # 500-D
#X = skdata.fetch_openml(name='AP_Endometrium_Breast', version=1)           # 400-D
#X = skdata.fetch_openml(name='AP_Ovary_Lung', version=1)                   # 100-D
#X = skdata.fetch_openml(name='OVA_Uterus', version=1)                      # 100-D
#X = skdata.fetch_openml(name='micro-mass', version=1)                      # 100-D
#X = skdata.fetch_openml(name='har', version=1)                             # 10%  of the samples and 100-D       
#X = skdata.fetch_openml(name='eating', version=1)                          # 100-D       
#X = skdata.fetch_openml(name='oh5.wc', version=1)                          # 40-D
#X = skdata.fetch_openml(name='leukemia', version=1)                        # 40-D


dados = X['data']
target = X['target']

# Convert labels to integers
lista = []
for x in target:
    if x not in lista:  
        lista.append(x)     
# Map labels to respective numbers
rotulos = []
for x in target:  
    for i in range(len(lista)):
        if x == lista[i]:  
            rotulos.append(i)
target = np.array(rotulos)

# Number of samples, features and classes
n = dados.shape[0]
m = dados.shape[1]
c = len(np.unique(target))

# Some adjustments are require in opnML datasets
# Categorical features must be encoded manually
if type(dados) == sp.sparse._csr.csr_matrix:
    dados = dados.todense()
    dados = np.asarray(dados)

if not isinstance(dados, np.ndarray):
    cat_cols = dados.select_dtypes(['category']).columns
    dados[cat_cols] = dados[cat_cols].apply(lambda x: x.cat.codes)
    # Convert to numpy
    dados = dados.to_numpy()

# To remove NaNs
dados = np.nan_to_num(dados)

# Data standardization (to deal with variables having different units/scales)
dados = preprocessing.scale(dados)

# OPTIONAL: set this flag to True to reduce the number of samples
reduce_samples = False
reduce_dim = False

if not reduce_samples and not reduce_dim:
    raw_data = dados

if reduce_samples:
    percentage = 0.25
    dados, garbage, target, garbage_t = train_test_split(dados, target, train_size=percentage, random_state=42)
    raw_data = dados

# OPTIONAL: set this flag to True to reduce the dimensionality with PCA prior to metric learning
if reduce_dim:
    num_features = 50
    raw_data = dados
    dados = PCA(n_components=num_features).fit_transform(dados)

# Number of samples, features and classes
n = dados.shape[0]
m = dados.shape[1]

# Print data info
print('N = ', n)        
print('M = ', m)        
print('C = %d' %c)      
# Number of neighbors in KNN graph (patch size)
nn = round(sqrt(n))                 

# GMM algorithm
CLUSTER = 'gmm'


############## K-ISOMAP 
# Number of neighbors
print('K = ', nn)      
print()
print('Press enter to continue...')
input()


# Computes the results for all 10 curvature based metrics
lista_ri = []
lista_ch = []
lista_fm = []
lista_v = []
inicio = time.time()
for i in range(11):
    dados_kiso = KIsomap(dados, nn, 2, i)       
    L_kiso = Clustering(dados_kiso.T, target, 'K-ISOMAP', CLUSTER)
    lista_ri.append(L_kiso[0])
    lista_ch.append(L_kiso[1])
    lista_fm.append(L_kiso[2])
    lista_v.append(L_kiso[3])
    labels_kiso = L_kiso[4]
fim = time.time()
print('K-ISOMAP time: %f s' %(fim - inicio))
print()

# Find best result in terms of Rand index
print('*********************************************')
print('******* SUMMARY OF THE RESULTS **************')
print('*********************************************')
print()
print('Best K-ISOMAP result in terms of Rand index')
print('----------------------------------------------')
ri_star = max(enumerate(lista_ri), key=lambda x: x[1])[0]
dados_kiso = KIsomap(dados, nn, 2, ri_star) 
L_kiso = Clustering(dados_kiso.T, target, 'K-ISOMAP', CLUSTER)
labels_kiso = L_kiso[4]
PlotaDados(dados_kiso, labels_kiso, 'K-ISOMAP RI')

print('Best K-ISOMAP result in terms of Fowlkes Mallows')
print('----------------------------------------------')
fm_star = max(enumerate(lista_fm), key=lambda x: x[1])[0]
dados_kiso = KIsomap(dados, nn, 2, fm_star) 
L_kiso = Clustering(dados_kiso.T, target, 'K-ISOMAP', CLUSTER)
labels_kiso = L_kiso[4]
PlotaDados(dados_kiso, labels_kiso, 'K-ISOMAP FM')

# Find best result in terms of Rand index
print('Best K-ISOMAP result in terms of Calinski-Harabasz')
print('-----------------------------------------------------')
ch_star = max(enumerate(lista_ch), key=lambda x: x[1])[0]
dados_kiso = KIsomap(dados, nn, 2, ch_star) 
L_kiso = Clustering(dados_kiso.T, target, 'K-ISOMAP', CLUSTER)
labels_kiso = L_kiso[4]
PlotaDados(dados_kiso, labels_kiso, 'K-ISOMAP CH')

# Find best result in terms of V measure
print('Best K-ISOMAP result in terms of V measure')
print('-----------------------------------------------------')
v_star = max(enumerate(lista_v), key=lambda x: x[1])[0]
dados_kiso = KIsomap(dados, nn, 2, v_star) 
L_kiso = Clustering(dados_kiso.T, target, 'K-ISOMAP', CLUSTER)
labels_kiso = L_kiso[4]
PlotaDados(dados_kiso, labels_kiso, 'K-ISOMAP VS')

############## Regular ISOMAP 
print('ISOMAP result')
print('---------------')
model = Isomap(n_neighbors=nn, n_components=2)
dados_isomap = model.fit_transform(dados)
dados_isomap = dados_isomap.T
L_iso = Clustering(dados_isomap, target, 'ISOMAP', CLUSTER)
labels_iso = L_iso[4]
PlotaDados(dados_isomap.T, labels_iso, 'ISOMAP')

############## RAW DATA
print('RAW DATA result')
print('-----------------')
L_ = Clustering(raw_data.T, target, 'RAW DATA', CLUSTER)
labels_ = L_[4]