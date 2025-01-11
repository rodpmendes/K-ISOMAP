#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Unsupervised metric learning via K-ISOMAP for high-dimensional data clustering

Created on Wed Jul 24 16:59:33 2023


Add configuration to run set of noise experiments with gaussian, salt and pepper and poison noise
Add UMAP comparison

Updated on Wed May 15 19:56:14 2024



"""

# Imports
import time
import warnings
import numpy as np
import scipy as sp
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import sklearn.datasets as skdata
import sklearn.neighbors as sknn
from numpy import sqrt
from numpy.linalg import norm
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import train_test_split
from sklearn.metrics.cluster import rand_score
from sklearn.metrics.cluster import v_measure_score
from sklearn.metrics.cluster import fowlkes_mallows_score
from sklearn.metrics.cluster import calinski_harabasz_score
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics.pairwise import pairwise_distances
from umap import UMAP                 # install with: pip install umap
import json
import sys
from skimage.util import random_noise
from functions import HopfLink, RepliclustArchetype
from scipy.sparse.csgraph import connected_components
from sklearn.decomposition import KernelPCA
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import SpectralEmbedding
from sklearn.manifold import TSNE

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

    # Verificar o número de componentes conectados
    n_connected_components, components_labels = connected_components(knnGraph)

    # Caso o número de componentes conectados seja maior que 1
    if n_connected_components > 1:

        # Corrigir componentes conexas
        nbg = FixComponents(
            X=dados,
            graph=knnGraph,
            n_connected_components=n_connected_components,
            component_labels=components_labels
        )

        A = nbg.toarray()

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
            Wpca = maiores_autovetores  # Eigenvectors in columns
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
                    B[i, j] = delta[0]                     # metric A1 - Curvature of the first principal component
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

        # Verificar o número de componentes conectados
    n_connected_components, components_labels = connected_components(B)

    # Caso o número de componentes conectados seja maior que 1
    if n_connected_components > 1:

        # Corrigir componentes conexas
        B = FixComponents(
            X=B,
            graph=B,
            n_connected_components=n_connected_components,
            component_labels=components_labels
        )
                
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
    

def FixComponents(
    X,
    graph,
    n_connected_components,
    component_labels):

    for i in range(n_connected_components):
        idx_i = np.flatnonzero(component_labels == i)
        Xi = X[idx_i]

        for j in range(i):
            idx_j = np.flatnonzero(component_labels == j)
            Xj = X[idx_j]

            D = pairwise_distances(Xi, Xj, metric="euclidean")

            ii, jj = np.unravel_index(D.argmin(axis=None), D.shape)

            graph[idx_i[ii], idx_j[jj]] = D[ii, jj]
            graph[idx_j[jj], idx_i[ii]] = D[ii, jj]

    return graph


'''
 Performs clustering in data, returns the obtained labels and evaluates the clusters
'''
def Clustering(dados, target, DR_method, cluster):
    rand, ca, fm, v, dbs, ss = -1, -1, -1, -1, -1, -1
    labels = [-1,-2]
    try:
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
        dbs = davies_bouldin_score(dados.T, labels)
        ss = silhouette_score(dados.T, labels)
        # Print evaluation metrics
        print('Rand index: ', rand)    
        print('Calinski Harabasz: ', ca)
        print('Fowlkes Mallows:', fm)
        print('V measure:', v)
        print('Davies Bouldin Score:', dbs)
        print('Silhouette Score:', ss)
        print()

    except Exception as e:
        print(DR_method + " -------- def Clustering error:", e)
    finally:
        return [np.float64(rand), np.float64(ca), np.float64(fm), np.float64(v), np.float64(dbs), np.float64(ss), labels.tolist()]



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
    nome_arquivo = metodo + '.jpeg'
    plt.title(metodo+' clusters')
    plt.savefig(nome_arquivo, dpi=300)
    plt.close()

def MakeSyntheticData():

    X1, y1 = HopfLink(noise=True)
    X2, y2 = RepliclustArchetype()

    hopf_link = {'data': X1, 'target': y1, 'details': {'name':'Hopf-Link'}}
    repliclust_archetype = {'data': X2, 'target': y2, 'details': {'name':'Repliclust-Archetype'}}
    
    return hopf_link, repliclust_archetype


##################### Beginning of thescript

#####################  Data loading
def main():

    hopf_link, repliclust_archetype = MakeSyntheticData()
    
    #To perform the experiments according to the article, uncomment the desired sets of datasets
    datasets = [
        # Synthetic data
         {"db": hopf_link, "reduce_samples": False, "percentage":0, "reduce_dim":False, "num_features": 0},
         {"db": repliclust_archetype, "reduce_samples": False, "percentage":0, "reduce_dim":False, "num_features": 0}

        # First set of experiments
        # {"db": skdata.fetch_openml(name='servo', version=1), "reduce_samples": False, "percentage":0, "reduce_dim":False, "num_features": 0},
        # {"db": skdata.fetch_openml(name='car-evaluation', version=1), "reduce_samples": False, "percentage":0, "reduce_dim":False, "num_features": 0},
        # {"db": skdata.fetch_openml(name='breast-tissue', version=2), "reduce_samples": False, "percentage":0, "reduce_dim":False, "num_features": 0},
        # {"db": skdata.fetch_openml(name='Engine1', version=1), "reduce_samples": False, "percentage":0, "reduce_dim":False, "num_features": 0},
        # {"db": skdata.fetch_openml(name='xd6', version=1), "reduce_samples": False, "percentage":0, "reduce_dim":False, "num_features": 0},
        # {"db": skdata.fetch_openml(name='heart-h', version=3), "reduce_samples": False, "percentage":0, "reduce_dim":False, "num_features": 0},
        # {"db": skdata.fetch_openml(name='steel-plates-fault', version=3), "reduce_samples": False, "percentage":0, "reduce_dim":False, "num_features": 0},
        # {"db": skdata.fetch_openml(name='PhishingWebsites', version=1), "reduce_samples": True, "percentage":.1, "reduce_dim":False, "num_features": 0},              # 10% of the samples
        # {"db": skdata.fetch_openml(name='satimage', version=1), "reduce_samples": True, "percentage":.25, "reduce_dim":False, "num_features": 0},                     # 25% of the samples
        # {"db": skdata.fetch_openml(name='led24', version=1), "reduce_samples": True, "percentage":.20, "reduce_dim":False, "num_features": 0},                       # 20% of the samples
        # {"db": skdata.fetch_openml(name='hayes-roth', version=2), "reduce_samples": False, "percentage":0, "reduce_dim":False, "num_features": 0},
        # {"db": skdata.fetch_openml(name='rabe_131', version=2), "reduce_samples": False, "percentage":0, "reduce_dim":False, "num_features": 0},
        # {"db": skdata.fetch_openml(name='prnn_synth', version=1), "reduce_samples": False, "percentage":0, "reduce_dim":False, "num_features": 0},
        # {"db": skdata.fetch_openml(name='visualizing_environmental', version=2), "reduce_samples": False, "percentage":0, "reduce_dim":False, "num_features": 0},
        # {"db": skdata.fetch_openml(name='diggle_table_a2', version=2), "reduce_samples": False, "percentage":0, "reduce_dim":False, "num_features": 0},
        # {"db": skdata.fetch_openml(name='newton_hema', version=2), "reduce_samples": False, "percentage":0, "reduce_dim":False, "num_features": 0},
        # {"db": skdata.fetch_openml(name='wisconsin', version=2), "reduce_samples": False, "percentage":0, "reduce_dim":False, "num_features": 0},
        # {"db": skdata.fetch_openml(name='fri_c4_250_100', version=2), "reduce_samples": False, "percentage":0, "reduce_dim":False, "num_features": 0},
        # {"db": skdata.fetch_openml(name='conference_attendance', version=1), "reduce_samples": False, "percentage":0, "reduce_dim":False, "num_features": 0},
        # {"db": skdata.fetch_openml(name='tic-tac-toe', version=1), "reduce_samples": False, "percentage":0, "reduce_dim":False, "num_features": 0},
        # {"db": skdata.fetch_openml(name='qsar-biodeg', version=1), "reduce_samples": False, "percentage":0, "reduce_dim":False, "num_features": 0},
        # {"db": skdata.fetch_openml(name='spambase', version=1), "reduce_samples": True, "percentage":.25, "reduce_dim":False, "num_features": 0},                     # 25% of the samples
        # {"db": skdata.fetch_openml(name='cmc', version=1), "reduce_samples": False, "percentage":0, "reduce_dim":False, "num_features": 0},
        # {"db": skdata.fetch_openml(name='heart-statlog', version=1), "reduce_samples": False, "percentage":0, "reduce_dim":False, "num_features": 0},
        
        
        # Second set of experiments
        # {"db": skdata.fetch_openml(name='cnae-9', version=1), "reduce_samples": False, "percentage":0, "reduce_dim":True, "num_features": 50},                     # 50-D
        # {"db": skdata.fetch_openml(name='AP_Breast_Kidney', version=1), "reduce_samples": False, "percentage":0, "reduce_dim":True, "num_features": 500},          # 500-D
        # {"db": skdata.fetch_openml(name='AP_Endometrium_Breast', version=1), "reduce_samples": False, "percentage":0, "reduce_dim":True, "num_features": 400},     # 400-D
        # {"db": skdata.fetch_openml(name='AP_Ovary_Lung', version=1), "reduce_samples": False, "percentage":0, "reduce_dim":True, "num_features": 100},             # 100-D
        # {"db": skdata.fetch_openml(name='OVA_Uterus', version=1), "reduce_samples": False, "percentage":0, "reduce_dim":True, "num_features": 100},                # 100-D
        # {"db": skdata.fetch_openml(name='micro-mass', version=1), "reduce_samples": False, "percentage":0, "reduce_dim":True, "num_features": 100},                # 100-D
        # {"db": skdata.fetch_openml(name='har', version=1), "reduce_samples": True, "percentage":0.1, "reduce_dim":True, "num_features": 100},                      # 10%  of the samples and 100-D
        # {"db": skdata.fetch_openml(name='eating', version=1), "reduce_samples": False, "percentage":0, "reduce_dim":True, "num_features": 100},                    # 100-D
        # {"db": skdata.fetch_openml(name='oh5.wc', version=1), "reduce_samples": False, "percentage":0, "reduce_dim":True, "num_features": 40},                     # 40-D
        # {"db": skdata.fetch_openml(name='leukemia', version=1), "reduce_samples": False, "percentage":0, "reduce_dim":True, "num_features": 40},                   # 40-D
         
         # Third set of datasets
         # {"db": skdata.fetch_openml(name='pendigits', version=1), "reduce_samples": True, "percentage":.25, "reduce_dim":False, "num_features": 0},
         # {"db": skdata.fetch_openml(name='COIL2000-train', version=1), "reduce_samples": True, "percentage":.25, "reduce_dim":False, "num_features": 0},
         ## {"db": skdata.fetch_openml(name='mnist_784', version=1), "reduce_samples": False, "percentage":0, "reduce_dim":True, "num_features": 100},
         ## {"db": skdata.fetch_openml(name='Fashion-MNIST', version=1), "reduce_samples": False, "percentage":0, "reduce_dim":True, "num_features": 100},
         
         # Fourth set of datasets
        
    ]
    
    plot_results = False
    apply_noise = False
    
    # ['gaussian', 'salt_pepper', 'poison']
    noise_type = 'none'
    
    if apply_noise:
        # Noise parameters
        # Standard deviation (spread or “width”) of the distribution. Must be non-negative
        data_std_dev = 1 # for normalized data base
        
        # Define magnitude
        magnitude = np.linspace(0, data_std_dev, 11)
    else:
        magnitude = np.linspace(0, 0, 1)
    
    # GMM algorithm
    # ['gmm', 'kmeans', 'agglomerative']
    CLUSTER = 'gmm'
    
    # File result
    file_results = 'third_dataset_results_v2.json'
    results = {}

    for dataset in datasets:
        
        X = dataset["db"]
        raw_data = X['data']
        dataset_data = X['data']
        dataset_target = X['target']
        dataset_name = X['details']['name']

        # Convert labels to integers
        label_list = []
        for x in dataset_target:
            if x not in label_list:  
                label_list.append(x)     
                
        # Map labels to respective numbers
        labels = []
        for x in dataset_target:  
            for i in range(len(label_list)):
                if x == label_list[i]:  
                    labels.append(i)
        dataset_target = np.array(labels)

        # Some adjustments are require in opnML datasets
        # Categorical features must be encoded manually
        if type(dataset_data) == sp.sparse._csr.csr_matrix:
            dataset_data = dataset_data.todense()
            dataset_data = np.asarray(dataset_data)

        if not isinstance(dataset_data, np.ndarray):
            cat_cols = dataset_data.select_dtypes(['category']).columns
            dataset_data[cat_cols] = dataset_data[cat_cols].apply(lambda x: x.cat.codes)
            # Convert to numpy
            dataset_data = dataset_data.to_numpy()

        # To remove NaNs
        dataset_data = np.nan_to_num(dataset_data)

        # Data standardization (to deal with variables having different units/scales)
        dataset_data = preprocessing.scale(dataset_data).astype(np.float64)

        # OPTIONAL: set this flag to True to reduce the number of samples
        reduce_samples = dataset["reduce_samples"]
        reduce_dim = dataset["reduce_dim"]

        if not reduce_samples and not reduce_dim:
            raw_data = dataset_data

        if reduce_samples:
            percentage = dataset["percentage"]
            dataset_data, garbage, dataset_target, garbage_t = train_test_split(dataset_data, dataset_target, train_size=percentage, random_state=42)
            raw_data = dataset_data

        # OPTIONAL: set this flag to True to reduce the dimensionality with PCA prior to metric learning
        if reduce_dim:
            num_features = dataset["num_features"]
            raw_data = dataset_data
            dataset_data = PCA(n_components=num_features).fit_transform(dataset_data)

        # Number of samples, features and classes
        n = dataset_data.shape[0]
        m = dataset_data.shape[1]
        c = len(np.unique(dataset_target))

        # Print data info
        print('N = ', n)
        print('M = ', m)
        print('C = %d' %c)
        
        # Number of neighbors in KNN graph (patch size)
        nn = round(sqrt(n))                 

        
        ############## K-ISOMAP 
        # Number of neighbors
        #print('K = ', nn)      
        #print()
        #print('Press enter to continue...')
        #input()


        # K-ISOMAP results
        ri_kiso, ch_kiso, fm_kiso, v_kiso, dbs_kiso, ss_kiso = [], [], [], [], [], []
        ch_kiso_norm, ri_kiso_norm, fm_kiso_norm, v_kiso_norm, dbs_kiso_norm, ss_kiso_norm = [], [], [], [], [], []
        ri_best_metric, ch_best_metric, fm_best_metric, v_best_metric, dbs_best_metric, ss_best_metric = [], [], [], [], [], []
        
        # ISOMAP results
        ri_iso, ch_iso, fm_iso, v_iso, dbs_iso, ss_iso = [], [], [], [], [], []
        ri_iso_norm, ch_iso_norm, fm_iso_norm, v_iso_norm, dbs_iso_norm, ss_iso_norm = [], [], [], [], [], []
        
        # UMAP results
        ri_umap, ch_umap, fm_umap, v_umap, dbs_umap, ss_umap = [], [], [], [], [], []
        ri_umap_norm, ch_umap_norm, fm_umap_norm, v_umap_norm, dbs_umap_norm, ss_umap_norm = [], [], [], [], [], []
        
        # RAW results
        ri_raw, ch_raw, fm_raw, v_raw, dbs_raw, ss_raw = [], [], [], [], [], []
        ri_raw_norm, ch_raw_norm, fm_raw_norm, v_raw_norm, dbs_raw_norm, ss_raw_norm = [], [], [], [], [], []
        
        # KernelPCA results
        ri_kpca, ch_kpca, fm_kpca, v_kpca, dbs_kpca, ss_kpca = [], [], [], [], [], []
        ri_kpca_norm, ch_kpca_norm, fm_kpca_norm, v_kpca_norm, dbs_kpca_norm, ss_kpca_norm = [], [], [], [], [], []
                
        # LocallyLinearEmbedding results
        ri_lle, ch_lle, fm_lle, v_lle, dbs_lle, ss_lle = [], [], [], [], [], []
        ri_lle_norm, ch_lle_norm, fm_lle_norm, v_lle_norm, dbs_lle_norm, ss_lle_norm = [], [], [], [], [], []
        
        # SpectralEmbedding results
        ri_se, ch_se, fm_se, v_se, dbs_se, ss_se = [], [], [], [], [], []
        ri_se_norm, ch_se_norm, fm_se_norm, v_se_norm, dbs_se_norm, ss_se_norm = [], [], [], [], [], []
        
        # TSNE results
        ri_tsne, ch_tsne, fm_tsne, v_tsne, dbs_tsne, ss_tsne = [], [], [], [], [], []
        ri_tsne_norm, ch_tsne_norm, fm_tsne_norm, v_tsne_norm, dbs_tsne_norm, ss_tsne_norm = [], [], [], [], [], []
            
        for r in range(len(magnitude)):
            # Computes the results for all 10 curvature based metrics
            start = time.time()
            
            if apply_noise:
                dataset_data = apply_noise_type(noise_type, dataset_data, magnitude[r])
                raw_data = apply_noise_type(noise_type, dataset_data, magnitude[r])
            
            ri, ch, fm, v, dbs, ss = [], [], [], [], [], []
            for i in range(11):
                DR_method = 'K-ISOMAP ' + dataset_name + ' option=' + str(i) + ' cluster=' + CLUSTER + ' mag=' + str(r)
                
                try:
                    dados_kiso = KIsomap(dataset_data, nn, 2, i)       
                except Exception as e:
                    print(DR_method + " -------- def KIsomap error:", e)
                    dados_kiso = []
                    
                if dados_kiso.any():
                    L_kiso = Clustering(dados_kiso.T, dataset_target, DR_method, CLUSTER)
                    ri.append(L_kiso[0])
                    ch.append(L_kiso[1])
                    fm.append(L_kiso[2])
                    v.append(L_kiso[3])
                    dbs.append(L_kiso[4])
                    ss.append(L_kiso[5])
            finish = time.time()
            print(dataset_name + ' K-ISOMAP time: %f s' %(finish - start))
            print()
            
            # Find best result in terms of Rand index of metric function
            ri_star = max(ri)
            ri_kiso.append(ri_star)
            ri_best_metric.append(ri.index(ri_star))
                        
            # Find best result in terms of Calinski Harabasz Score of metric function
            ch_star = max(ch)
            ch_kiso.append(ch_star)
            ch_best_metric.append(ch.index(ch_star))
            
            # Find best result in terms of Fowlkes Mallows Score of metric function
            fm_star = max(fm)
            fm_kiso.append(fm_star)
            fm_best_metric.append(fm.index(fm_star))
            
            # Find best result in terms of V measure of metric function
            v_star = max(v)
            v_kiso.append(v_star)
            v_best_metric.append(v.index(v_star))
            
            # Find best result in terms of Davies Bouldin Score
            dbs_star = max(dbs)
            dbs_kiso.append(dbs_star)
            dbs_best_metric.append(dbs.index(dbs_star))
            
            # Find best result in terms of Silhouette Score
            ss_star = max(ss)
            ss_kiso.append(ss_star)
            ss_best_metric.append(ss.index(ss_star))
            
            ############## Regular ISOMAP 
            print(dataset_name + ' ISOMAP result')
            print('---------------')
            model = Isomap(n_neighbors=nn, n_components=2)
            isomap_data = model.fit_transform(dataset_data)
            isomap_data = isomap_data.T
            DR_method = 'ISOMAP ' + dataset_name + ' cluster=' + CLUSTER
            L_iso = Clustering(isomap_data, dataset_target, DR_method, CLUSTER)
            ri_iso.append(L_iso[0])
            ch_iso.append(L_iso[1])
            fm_iso.append(L_iso[2])
            v_iso.append(L_iso[3])
            dbs_iso.append(L_iso[4])
            ss_iso.append(L_iso[5])


            ############## UMAP
            print(dataset_name + ' UMAP result')
            print('---------------')
            model = UMAP(n_components=2)
            umap_data = model.fit_transform(dataset_data)
            umap_data = umap_data.T
            DR_method = 'UMAP ' + dataset_name + ' cluster=' + CLUSTER
            L_umap = Clustering(umap_data, dataset_target, DR_method, CLUSTER)
            ri_umap.append(L_umap[0])
            ch_umap.append(L_umap[1])
            fm_umap.append(L_umap[2])
            v_umap.append(L_umap[3])
            dbs_umap.append(L_umap[4])
            ss_umap.append(L_umap[5])


            ############## RAW DATA
            print(dataset_name + ' RAW DATA result')
            print('-----------------')
            DR_method = 'RAW ' + dataset_name + ' cluster=' + CLUSTER
            if reduce_dim:
                L_ = Clustering(dataset_data.T, dataset_target, DR_method, CLUSTER)
            else:
                L_ = Clustering(raw_data.T, dataset_target, DR_method, CLUSTER)
            ri_raw.append(L_[0])
            ch_raw.append(L_[1])
            fm_raw.append(L_[2])
            v_raw.append(L_[3])
            dbs_raw.append(L_[4])
            ss_raw.append(L_[5])
            
            
            ############## KernelPCA
            print(dataset_name + ' KernelPCA result')
            print('---------------')
            model = KernelPCA(n_components=2)
            kpca_data = model.fit_transform(dataset_data)
            kpca_data = kpca_data.T
            DR_method = 'KernelPCA ' + dataset_name + ' cluster=' + CLUSTER
            L_kpca = Clustering(kpca_data, dataset_target, DR_method, CLUSTER)
            ri_kpca.append(L_kpca[0])
            ch_kpca.append(L_kpca[1])
            fm_kpca.append(L_kpca[2])
            v_kpca.append(L_kpca[3])
            dbs_kpca.append(L_kpca[4])
            ss_kpca.append(L_kpca[5])
            
            
            ############## LLE LocallyLinearEmbedding
            print(dataset_name + ' LocallyLinearEmbedding result')
            print('---------------')
            model = LocallyLinearEmbedding(n_components=2)
            lle_data = model.fit_transform(dataset_data)
            lle_data = lle_data.T
            DR_method = 'LocallyLinearEmbedding ' + dataset_name + ' cluster=' + CLUSTER
            L_lle = Clustering(lle_data, dataset_target, DR_method, CLUSTER)
            ri_lle.append(L_lle[0])
            ch_lle.append(L_lle[1])
            fm_lle.append(L_lle[2])
            v_lle.append(L_lle[3])
            dbs_lle.append(L_lle[4])
            ss_lle.append(L_lle[5])
            
            
            ############## SpectralEmbedding
            print(dataset_name + ' SpectralEmbedding result')
            print('---------------')
            model = SpectralEmbedding(n_components=2)
            se_data = model.fit_transform(dataset_data)
            se_data = se_data.T
            DR_method = 'SpectralEmbedding ' + dataset_name + ' cluster=' + CLUSTER
            L_se = Clustering(se_data, dataset_target, DR_method, CLUSTER)
            ri_se.append(L_se[0])
            ch_se.append(L_se[1])
            fm_se.append(L_se[2])
            v_se.append(L_se[3])
            dbs_se.append(L_se[4])
            ss_se.append(L_se[5])
            
            
            ############## T-SNE
            print(dataset_name + ' T-SNE result')
            print('---------------')
            model = TSNE(n_components=2)
            tsne_data = model.fit_transform(dataset_data)
            tsne_data = tsne_data.T
            DR_method = 'T-SNE ' + dataset_name + ' cluster=' + CLUSTER
            L_tsne = Clustering(tsne_data, dataset_target, DR_method, CLUSTER)
            ri_tsne.append(L_tsne[0])
            ch_tsne.append(L_tsne[1])
            fm_tsne.append(L_tsne[2])
            v_tsne.append(L_tsne[3])
            dbs_tsne.append(L_tsne[4])
            ss_tsne.append(L_tsne[5])

            
        results[dataset_name] = { "KISOMAP": [ri_kiso, ch_kiso, fm_kiso, v_kiso, dbs_kiso, ss_kiso],
                                  "ISOMAP": [ri_iso, ch_iso, fm_iso, v_iso, dbs_iso, ss_iso],
                                  "UMAP": [ri_umap, ch_umap, fm_umap, v_umap, dbs_umap, ss_umap],
                                  "RAW": [ri_raw, ch_raw, fm_raw, v_raw, dbs_raw, ss_raw],
                                  "KPCA": [ri_kpca, ch_kpca, fm_kpca, v_kpca, dbs_kpca, ss_kpca],
                                  "LLE": [ri_lle, ch_lle, fm_lle, v_lle, dbs_lle, ss_lle],
                                  "SE": [ri_se, ch_se, fm_se, v_se, dbs_se, ss_se],
                                  "TSNE": [ri_tsne, ch_tsne, fm_tsne, v_tsne, dbs_tsne, ss_tsne] 
                                  }
        
        
        
        # normalize data results
        ri_data = ri_kiso + ri_iso + ri_umap + ri_raw + ri_kpca + ri_lle + ri_se + ri_tsne
        min1, max1 = np.min(ri_data), np.max(ri_data)
        ri_data_normalized = (ri_data - min1) / (max1 - min1)
        ri_kiso_norm = ri_data_normalized[:len(ri_kiso)].tolist()
        ri_iso_norm = ri_data_normalized[len(ri_kiso):len(ri_kiso)+len(ri_iso)].tolist()
        ri_umap_norm = ri_data_normalized[len(ri_kiso)+len(ri_iso):len(ri_kiso)+len(ri_iso)+len(ri_umap)].tolist()
        ri_raw_norm = ri_data_normalized[len(ri_kiso)+len(ri_iso)+len(ri_umap):len(ri_kiso)+len(ri_iso)+len(ri_umap)+len(ri_raw)].tolist()
        ri_kpca_norm = ri_data_normalized[len(ri_kiso)+len(ri_iso)+len(ri_umap)+len(ri_raw):len(ri_kiso)+len(ri_iso)+len(ri_umap)+len(ri_raw)+len(ri_kpca)].tolist()
        ri_lle_norm = ri_data_normalized[len(ri_kiso)+len(ri_iso)+len(ri_umap)+len(ri_raw)+len(ri_kpca):len(ri_kiso)+len(ri_iso)+len(ri_umap)+len(ri_raw)+len(ri_kpca)+len(ri_lle)].tolist()
        ri_se_norm = ri_data_normalized[len(ri_kiso)+len(ri_iso)+len(ri_umap)+len(ri_raw)+len(ri_kpca)+len(ri_lle):len(ri_kiso)+len(ri_iso)+len(ri_umap)+len(ri_raw)+len(ri_kpca)+len(ri_lle)+len(ri_se)].tolist()
        ri_tsne_norm = ri_data_normalized[len(ri_kiso)+len(ri_iso)+len(ri_umap)+len(ri_raw)+len(ri_kpca)+len(ri_lle)+len(ri_se):len(ri_kiso)+len(ri_iso)+len(ri_umap)+len(ri_raw)+len(ri_kpca)+len(ri_lle)+len(ri_se)+len(ri_tsne)].tolist()
        
        
        ch_data = ch_kiso + ch_iso + ch_umap + ch_raw + ch_kpca + ch_lle + ch_se + ch_tsne
        min1, max1 = np.min(ch_data), np.max(ch_data)
        ch_data_normalized = (ch_data - min1) / (max1 - min1)
        ch_kiso_norm = ch_data_normalized[:len(ch_kiso)].tolist()
        ch_iso_norm = ch_data_normalized[len(ch_kiso):len(ch_kiso)+len(ch_iso)].tolist()
        ch_umap_norm = ch_data_normalized[len(ch_kiso)+len(ch_iso):len(ch_kiso)+len(ch_iso)+len(ch_umap)].tolist()
        ch_raw_norm = ch_data_normalized[len(ch_kiso)+len(ch_iso)+len(ch_umap):len(ch_kiso)+len(ch_iso)+len(ch_umap)+len(ch_raw)].tolist()
        ch_kpca_norm = ch_data_normalized[len(ch_kiso)+len(ch_iso)+len(ch_umap)+len(ch_raw):len(ch_kiso)+len(ch_iso)+len(ch_umap)+len(ch_raw)+len(ch_kpca)].tolist()
        ch_lle_norm = ch_data_normalized[len(ch_kiso)+len(ch_iso)+len(ch_umap)+len(ch_raw)+len(ch_kpca):len(ch_kiso)+len(ch_iso)+len(ch_umap)+len(ch_raw)+len(ch_kpca)+len(ch_lle)].tolist()
        ch_se_norm = ch_data_normalized[len(ch_kiso)+len(ch_iso)+len(ch_umap)+len(ch_raw)+len(ch_kpca)+len(ch_lle):len(ch_kiso)+len(ch_iso)+len(ch_umap)+len(ch_raw)+len(ch_kpca)+len(ch_lle)+len(ch_se)].tolist()
        ch_tsne_norm = ch_data_normalized[len(ch_kiso)+len(ch_iso)+len(ch_umap)+len(ch_raw)+len(ch_kpca)+len(ch_lle)+len(ch_se):len(ch_kiso)+len(ch_iso)+len(ch_umap)+len(ch_raw)+len(ch_kpca)+len(ch_lle)+len(ch_se)+len(ch_tsne)].tolist()
        
        
        fm_data = fm_kiso + fm_iso + fm_umap + fm_raw + fm_kpca + fm_lle + fm_se + fm_tsne
        min1, max1 = np.min(fm_data), np.max(fm_data)
        fm_data_normalized = (fm_data - min1) / (max1 - min1)
        fm_kiso_norm = fm_data_normalized[:len(fm_kiso)].tolist()
        fm_iso_norm = fm_data_normalized[len(fm_kiso):len(fm_kiso)+len(fm_iso)].tolist()
        fm_umap_norm = fm_data_normalized[len(fm_kiso)+len(fm_iso):len(fm_kiso)+len(fm_iso)+len(fm_umap)].tolist()
        fm_raw_norm = fm_data_normalized[len(fm_kiso)+len(fm_iso)+len(fm_umap):len(fm_kiso)+len(fm_iso)+len(fm_umap)+len(fm_raw)].tolist()
        fm_kpca_norm = fm_data_normalized[len(fm_kiso)+len(fm_iso)+len(fm_umap)+len(fm_raw):len(fm_kiso)+len(fm_iso)+len(fm_umap)+len(fm_raw)+len(fm_kpca)].tolist()
        fm_lle_norm = fm_data_normalized[len(fm_kiso)+len(fm_iso)+len(fm_umap)+len(fm_raw)+len(fm_kpca):len(fm_kiso)+len(fm_iso)+len(fm_umap)+len(fm_raw)+len(fm_kpca)+len(fm_lle)].tolist()
        fm_se_norm = fm_data_normalized[len(fm_kiso)+len(fm_iso)+len(fm_umap)+len(fm_raw)+len(fm_kpca)+len(fm_lle):len(fm_kiso)+len(fm_iso)+len(fm_umap)+len(fm_raw)+len(fm_kpca)+len(fm_lle)+len(fm_se)].tolist()
        fm_tsne_norm = fm_data_normalized[len(fm_kiso)+len(fm_iso)+len(fm_umap)+len(fm_raw)+len(fm_kpca)+len(fm_lle)+len(fm_se):len(fm_kiso)+len(fm_iso)+len(fm_umap)+len(fm_raw)+len(fm_kpca)+len(fm_lle)+len(fm_se)+len(fm_tsne)].tolist()

            
        v_data = v_kiso + v_iso + v_umap + v_raw + v_kpca + v_lle + v_se + v_tsne
        min1, max1 = np.min(v_data), np.max(v_data)
        v_data_normalized = (v_data - min1) / (max1 - min1)
        v_kiso_norm = v_data_normalized[:len(v_kiso)].tolist()
        v_iso_norm = v_data_normalized[len(v_kiso):len(v_kiso)+len(v_iso)].tolist()
        v_umap_norm = v_data_normalized[len(v_kiso)+len(v_iso):len(v_kiso)+len(v_iso)+len(v_umap)].tolist()
        v_raw_norm = v_data_normalized[len(v_kiso)+len(v_iso)+len(v_umap):len(v_kiso)+len(v_iso)+len(v_umap)+len(v_raw)].tolist()
        v_kpca_norm = v_data_normalized[len(v_kiso)+len(v_iso)+len(v_umap)+len(v_raw):len(v_kiso)+len(v_iso)+len(v_umap)+len(v_raw)+len(v_kpca)].tolist()
        v_lle_norm = v_data_normalized[len(v_kiso)+len(v_iso)+len(v_umap)+len(v_raw)+len(v_kpca):len(v_kiso)+len(v_iso)+len(v_umap)+len(v_raw)+len(v_kpca)+len(v_lle)].tolist()
        v_se_norm = v_data_normalized[len(v_kiso)+len(v_iso)+len(v_umap)+len(v_raw)+len(v_kpca)+len(v_lle):len(v_kiso)+len(v_iso)+len(v_umap)+len(v_raw)+len(v_kpca)+len(v_lle)+len(v_se)].tolist()
        v_tsne_norm = v_data_normalized[len(v_kiso)+len(v_iso)+len(v_umap)+len(v_raw)+len(v_kpca)+len(v_lle)+len(v_se):len(v_kiso)+len(v_iso)+len(v_umap)+len(v_raw)+len(v_kpca)+len(v_lle)+len(v_se)+len(v_tsne)].tolist()
                
        
        dbs_data = dbs_kiso + dbs_iso + dbs_umap + dbs_raw + dbs_kpca + dbs_lle + dbs_se + dbs_tsne
        min1, max1 = np.min(dbs_data), np.max(dbs_data)
        dbs_data_normalized = (dbs_data - min1) / (max1 - min1)
        dbs_kiso_norm = dbs_data_normalized[:len(dbs_kiso)].tolist()
        dbs_iso_norm = dbs_data_normalized[len(dbs_kiso):len(dbs_kiso)+len(dbs_iso)].tolist()
        dbs_umap_norm = dbs_data_normalized[len(dbs_kiso)+len(dbs_iso):len(dbs_kiso)+len(dbs_iso)+len(dbs_umap)].tolist()
        dbs_raw_norm = dbs_data_normalized[len(dbs_kiso)+len(dbs_iso)+len(dbs_umap):len(dbs_kiso)+len(dbs_iso)+len(dbs_umap)+len(dbs_raw)].tolist()
        dbs_kpca_norm = dbs_data_normalized[len(dbs_kiso)+len(dbs_iso)+len(dbs_umap)+len(dbs_raw):len(dbs_kiso)+len(dbs_iso)+len(dbs_umap)+len(dbs_raw)+len(dbs_kpca)].tolist()
        dbs_lle_norm = dbs_data_normalized[len(dbs_kiso)+len(dbs_iso)+len(dbs_umap)+len(dbs_raw)+len(dbs_kpca):len(dbs_kiso)+len(dbs_iso)+len(dbs_umap)+len(dbs_raw)+len(dbs_kpca)+len(dbs_lle)].tolist()
        dbs_se_norm = dbs_data_normalized[len(dbs_kiso)+len(dbs_iso)+len(dbs_umap)+len(dbs_raw)+len(dbs_kpca)+len(dbs_lle):len(dbs_kiso)+len(dbs_iso)+len(dbs_umap)+len(dbs_raw)+len(dbs_kpca)+len(dbs_lle)+len(dbs_se)].tolist()
        dbs_tsne_norm = dbs_data_normalized[len(dbs_kiso)+len(dbs_iso)+len(dbs_umap)+len(dbs_raw)+len(dbs_kpca)+len(dbs_lle)+len(dbs_se):len(dbs_kiso)+len(dbs_iso)+len(dbs_umap)+len(dbs_raw)+len(dbs_kpca)+len(dbs_lle)+len(dbs_se)+len(dbs_tsne)].tolist()
        
        
        ss_data = ss_kiso + ss_iso + ss_umap + ss_raw + ss_kpca + ss_lle + ss_se + ss_tsne
        min1, max1 = np.min(ss_data), np.max(ss_data)
        ss_data_normalized = (ss_data - min1) / (max1 - min1)
        ss_kiso_norm = ss_data_normalized[:len(ss_kiso)].tolist()
        ss_iso_norm = ss_data_normalized[len(ss_kiso):len(ss_kiso)+len(ss_iso)].tolist()
        ss_umap_norm = ss_data_normalized[len(ss_kiso)+len(ss_iso):len(ss_kiso)+len(ss_iso)+len(ss_umap)].tolist()
        ss_raw_norm = ss_data_normalized[len(ss_kiso)+len(ss_iso)+len(ss_umap):len(ss_kiso)+len(ss_iso)+len(ss_umap)+len(ss_raw)].tolist()
        ss_kpca_norm = ss_data_normalized[len(ss_kiso)+len(ss_iso)+len(ss_umap)+len(ss_raw):len(ss_kiso)+len(ss_iso)+len(ss_umap)+len(ss_raw)+len(ss_kpca)].tolist()
        ss_lle_norm = ss_data_normalized[len(ss_kiso)+len(ss_iso)+len(ss_umap)+len(ss_raw)+len(ss_kpca):len(ss_kiso)+len(ss_iso)+len(ss_umap)+len(ss_raw)+len(ss_kpca)+len(ss_lle)].tolist()
        ss_se_norm = ss_data_normalized[len(ss_kiso)+len(ss_iso)+len(ss_umap)+len(ss_raw)+len(ss_kpca)+len(ss_lle):len(ss_kiso)+len(ss_iso)+len(ss_umap)+len(ss_raw)+len(ss_kpca)+len(ss_lle)+len(ss_se)].tolist()
        ss_tsne_norm = ss_data_normalized[len(ss_kiso)+len(ss_iso)+len(ss_umap)+len(ss_raw)+len(ss_kpca)+len(ss_lle)+len(ss_se):len(ss_kiso)+len(ss_iso)+len(ss_umap)+len(ss_raw)+len(ss_kpca)+len(ss_lle)+len(ss_se)+len(ss_tsne)].tolist()

        
        results[dataset_name + '_norm'] = { "KISOMAP": [ri_kiso_norm, ch_kiso_norm, fm_kiso_norm, v_kiso_norm, dbs_kiso_norm, ss_kiso_norm],
                                            "ISOMAP": [ri_iso_norm, ch_iso_norm, fm_iso_norm, v_iso_norm, dbs_iso_norm, ss_iso_norm],
                                            "UMAP": [ri_umap_norm, ch_umap_norm, fm_umap_norm, v_umap_norm, dbs_umap_norm, ss_umap_norm],
                                            "RAW": [ri_raw_norm, ch_raw_norm, fm_raw_norm, v_raw_norm, dbs_raw_norm, ss_raw_norm],
                                            "KPCA": [ri_kpca_norm, ch_kpca_norm, fm_kpca_norm, v_kpca_norm, dbs_kpca_norm, ss_kpca_norm],
                                            "LLE": [ri_lle_norm, ch_lle_norm, fm_lle_norm, v_lle_norm, dbs_lle_norm, ss_lle_norm],
                                            "SE": [ri_se_norm, ch_se_norm, fm_se_norm, v_se_norm, dbs_se_norm, ss_se_norm],
                                            "TSNE": [ri_tsne_norm, ch_tsne_norm, fm_tsne_norm, v_tsne_norm, dbs_tsne_norm, ss_tsne_norm]}

        print('Dataset ', dataset_name,' complete')
        print()
        
        # Check previous results
        try:
            with open(file_results, 'r') as f:
                previous_results = json.load(f)
        except FileNotFoundError:
            previous_results = {}
                
        results = {key: {**results.get(key, {}), **previous_results.get(key, {})} for key in results.keys() | previous_results.keys()}

        # Save results
        try:
            with open(file_results, 'w') as f:
                json.dump(results, f)
        except IOError as e:
            print(f"An error occurred while writing to the file: {file_results} - {e}")
        
        
        
        
    
    if plot_results:
        print('*********************************************')
        print('******* SUMMARY OF THE RESULTS **************')
        print('*********************************************')
        print()

        # Find best result in terms of Rand index
        print('Best K-ISOMAP result in terms of Rand index')
        print('----------------------------------------------')
        ri_star = max(enumerate(ri_kiso), key=lambda x: x[1])[0]
        dados_kiso = KIsomap(dataset_data, nn, 2, ri_star) 
        L_kiso = Clustering(dados_kiso.T, dataset_target, 'K-ISOMAP', CLUSTER)
        labels_kiso = L_kiso[4]
        PlotaDados(dados_kiso, labels_kiso, dataset_name + ' K-ISOMAP RI')


        # Find best result in terms of Rand index
        print('Best K-ISOMAP result in terms of Calinski-Harabasz')
        print('-----------------------------------------------------')
        ch_star = max(enumerate(ch_kiso), key=lambda x: x[1])[0]
        dados_kiso = KIsomap(dataset_data, nn, 2, ch_star) 
        L_kiso = Clustering(dados_kiso.T, dataset_target, 'K-ISOMAP', CLUSTER)
        labels_kiso = L_kiso[4]
        PlotaDados(dados_kiso, labels_kiso, dataset_name + ' K-ISOMAP CH')


        # Find best result in terms of Fowlkes Mallows
        print('Best K-ISOMAP result in terms of Fowlkes Mallows')
        print('----------------------------------------------')
        fm_star = max(enumerate(fm_kiso), key=lambda x: x[1])[0]
        dados_kiso = KIsomap(dataset_data, nn, 2, fm_star) 
        L_kiso = Clustering(dados_kiso.T, dataset_target, 'K-ISOMAP', CLUSTER)
        labels_kiso = L_kiso[4]
        PlotaDados(dados_kiso, labels_kiso, dataset_name + dataset_name + ' K-ISOMAP FM')


        # Find best result in terms of V measure
        print('Best K-ISOMAP result in terms of V measure')
        print('-----------------------------------------------------')
        v_star = max(enumerate(v_kiso), key=lambda x: x[1])[0]
        dados_kiso = KIsomap(dataset_data, nn, 2, v_star) 
        L_kiso = Clustering(dados_kiso.T, dataset_target, 'K-ISOMAP', CLUSTER)
        labels_kiso = L_kiso[4]
        PlotaDados(dados_kiso, labels_kiso, dataset_name + ' K-ISOMAP VS')


        ############## Regular ISOMAP 
        print('ISOMAP result')
        print('---------------')
        model = Isomap(n_neighbors=nn, n_components=2)
        isomap_data = model.fit_transform(dataset_data)
        isomap_data = isomap_data.T
        L_iso = Clustering(isomap_data, dataset_target, 'ISOMAP', CLUSTER)
        labels_iso = L_iso[4]
        PlotaDados(isomap_data.T, labels_iso, dataset_name + ' ISOMAP')


        ############## UMAP
        print('UMAP result')
        print('---------------')
        model = UMAP(n_components=2)
        umap_data = model.fit_transform(dataset_data)
        umap_data = umap_data.T
        L_umap = Clustering(umap_data, dataset_target, 'UMAP', CLUSTER)
        labels_umap = L_umap[4]
        PlotaDados(umap_data.T, labels_umap, dataset_name + ' UMAP')

        ############## RAW DATA
        print('RAW DATA result')
        print('-----------------')
        L_ = Clustering(raw_data.T, dataset_target, 'RAW DATA', CLUSTER)
        labels_ = L_[4]
        #PlotaDados(raw_data.T, labels_, 'RAW DATA')
    
def gaussian(dataset, scale_mag):
    # Generate noise
    noise = np.random.normal(0, scale=scale_mag, size=dataset.shape)
    
    # Apply noise in feature level
    dataset_with_noise = dataset.copy() + noise
    
    return dataset_with_noise

def salt_pepper(dataset, scale_mag):
    # Apply noise salt and pepper
    noise = random_noise(dataset, mode='s&p', amount=scale_mag)
    
    # Apply noise in feature level
    dataset_with_noise = dataset.copy() + noise
    return dataset_with_noise

def poison(dataset, scale_mag):
    return 'poison'

def apply_noise_type(noise_type, dataset, scale_mag):
    switch = {
        'gaussian': gaussian,
        'salt_pepper': salt_pepper,
        'poision': poison
    }
    func = switch.get(noise_type, lambda: print("Invalid noise type"))
    
    return func(dataset, scale_mag)


if __name__ == '__main__':
    sys.exit(main())