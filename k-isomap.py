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
from umap import UMAP                 # install with: pip install umap
import json
import sys
from skimage.util import random_noise

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
    rand = ca = fm = v = -1
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
        # Print evaluation metrics
        print('Rand index: ', rand)    
        print('Calinski Harabasz: ', ca)
        print('Fowlkes Mallows:', fm)
        print('V measure:', v)
        print()

    except Exception as e:
        print(DR_method + " -------- def Clustering error:", e)
    finally:
        return [rand, ca, fm, v, labels.tolist()]



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


##################### Beginning of thescript

#####################  Data loading
def main():
    
    #To perform the experiments according to the article, uncomment the desired sets of datasets
    datasets = [
        # First set of experiments
        {"db": skdata.fetch_openml(name='servo', version=1), "reduce_samples": False, "percentage":0, "reduce_dim":False, "num_features": 0},
        # {"db": skdata.fetch_openml(name='car-evaluation', version=1), "reduce_samples": False, "percentage":0, "reduce_dim":False, "num_features": 0},
        # {"db": skdata.fetch_openml(name='breast-tissue', version=2), "reduce_samples": False, "percentage":0, "reduce_dim":False, "num_features": 0},
        # {"db": skdata.fetch_openml(name='Engine1', version=1), "reduce_samples": False, "percentage":0, "reduce_dim":False, "num_features": 0},
        # {"db": skdata.fetch_openml(name='xd6', version=1), "reduce_samples": False, "percentage":0, "reduce_dim":False, "num_features": 0},
        # {"db": skdata.fetch_openml(name='heart-h', version=3), "reduce_samples": False, "percentage":0, "reduce_dim":False, "num_features": 0},
        # {"db": skdata.fetch_openml(name='steel-plates-fault', version=3), "reduce_samples": False, "percentage":0, "reduce_dim":False, "num_features": 0},
        # {"db": skdata.fetch_openml(name='PhishingWebsites', version=1), "reduce_samples": True, "percentage":.1, "reduce_dim":False, "num_features": 0},              # 10% of the samples
        # {"db": skdata.fetch_openml(name='satimage', version=1), "reduce_samples": True, "percentage":.25, "reduce_dim":False, "num_features": 0},                     # 25% of the samples
        # {"db": skdata.fetch_openml(name='led24', version=1), "reduce_samples": True, "percentage":.20}, "reduce_dim":False, "num_features": 0,                        # 20% of the samples
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
    file_results = 'first_dataset_results.json'
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

        # Number of samples, features and classes
        n = dataset_data.shape[0]
        m = dataset_data.shape[1]
        c = len(np.unique(dataset_target))

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
        ri_kiso, ch_kiso, fm_kiso, v_kiso = [], [], [], []
        ch_kiso_norm, ri_kiso_norm, fm_kiso_norm, v_kiso_norm = [], [], [], []
        ri_best_metric, ch_best_metric, fm_best_metric, v_best_metric = [], [], [], []
        
        # ISOMAP results
        ri_iso, ch_iso, fm_iso, v_iso = [], [], [], []
        ri_iso_norm, ch_iso_norm, fm_iso_norm, v_iso_norm = [], [], [], []
        
        # UMAP results
        ri_umap, ch_umap, fm_umap, v_umap = [], [], [], []
        ri_umap_norm, ch_umap_norm, fm_umap_norm, v_umap_norm = [], [], [], []
        
        # RAW results
        ri_raw, ch_raw, fm_raw, v_raw = [], [], [], []
        ri_raw_norm, ch_raw_norm, fm_raw_norm, v_raw_norm = [], [], [], []
            
        for r in range(len(magnitude)):
            # Computes the results for all 10 curvature based metrics
            start = time.time()
            
            if apply_noise:
                dataset_data = apply_noise_type(noise_type, dataset_data, magnitude[r])
                raw_data = apply_noise_type(noise_type, dataset_data, magnitude[r])
            
            ri, ch, fm, v = [], [], [], []
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

            
        results[dataset_name] = { "KISOMAP": [ri_kiso, ch_kiso, fm_kiso, v_kiso],
                                  "ISOMAP": [ri_iso, ch_iso, fm_iso, v_iso],
                                  "UMAP": [ri_umap, ch_umap, fm_umap, v_umap],
                                  "RAW": [ri_raw, ch_raw, fm_raw, v_raw]}
        
        
        
        # normalize data results
        ri_data = ri_kiso + ri_iso + ri_umap + ri_raw
        min1, max1 = np.min(ri_data), np.max(ri_data)
        ri_data_normalized = (ri_data - min1) / (max1 - min1)
        ri_kiso_norm = ri_data_normalized[:len(ri_kiso)].tolist()
        ri_iso_norm = ri_data_normalized[len(ri_kiso):len(ri_kiso)+len(ri_iso)].tolist()
        ri_umap_norm = ri_data_normalized[len(ri_kiso)+len(ri_iso):len(ri_kiso)+len(ri_iso)+len(ri_umap)].tolist()
        ri_raw_norm = ri_data_normalized[len(ri_kiso)+len(ri_iso)+len(ri_umap):].tolist()
        
        
        ch_data = ch_kiso + ch_iso + ch_umap + ri_raw
        min1, max1 = np.min(ch_data), np.max(ch_data)
        ch_data_normalized = (ch_data - min1) / (max1 - min1)
        ch_kiso_norm = ch_data_normalized[:len(ch_kiso)].tolist()
        ch_iso_norm = ch_data_normalized[len(ch_kiso):len(ch_kiso)+len(ch_iso)].tolist()
        ch_umap_norm = ch_data_normalized[len(ch_kiso)+len(ch_iso):len(ch_kiso)+len(ch_iso)+len(ch_umap)].tolist()
        ch_raw_norm = ch_data_normalized[len(ch_kiso)+len(ch_iso)+len(ch_umap):].tolist()
        
        
        fm_data = fm_kiso + fm_iso + fm_umap + fm_raw
        min1, max1 = np.min(fm_data), np.max(fm_data)
        fm_data_normalized = (fm_data - min1) / (max1 - min1)
        fm_kiso_norm = fm_data_normalized[:len(fm_kiso)].tolist()
        fm_iso_norm = fm_data_normalized[len(fm_kiso):len(fm_kiso)+len(fm_iso)].tolist()
        fm_umap_norm = fm_data_normalized[len(fm_kiso)+len(fm_iso):len(fm_kiso)+len(fm_iso)+len(fm_umap)].tolist()
        fm_raw_norm = fm_data_normalized[len(fm_kiso)+len(fm_iso)+len(fm_umap):].tolist()
        
            
        v_data = v_kiso + v_iso + v_umap + v_raw
        min1, max1 = np.min(v_data), np.max(v_data)
        v_data_normalized = (v_data - min1) / (max1 - min1)
        v_kiso_norm = v_data_normalized[:len(v_kiso)].tolist()
        v_iso_norm = v_data_normalized[len(v_kiso):len(v_kiso)+len(v_iso)].tolist()
        v_umap_norm = v_data_normalized[len(v_kiso)+len(v_iso):len(v_kiso)+len(v_iso)+len(v_umap)].tolist()
        v_raw_norm = v_data_normalized[len(v_kiso)+len(v_iso)+len(v_umap):].tolist()

        
        results[dataset_name + '_norm'] = { "KISOMAP": [ri_kiso_norm, ch_kiso_norm, fm_kiso_norm, v_kiso_norm],
                                            "ISOMAP": [ri_iso_norm, ch_iso_norm, fm_iso_norm, v_iso_norm],
                                            "UMAP": [ri_umap_norm, ch_umap_norm, fm_umap_norm, v_umap_norm],
                                            "RAW": [ri_raw_norm, ch_raw_norm, fm_raw_norm, v_raw_norm]}
        
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