import json
import matplotlib.pyplot as plt
import numpy as np

file_results = 'first_dataset_results.json'

with open(file_results, 'r') as f:
    results = json.load(f)
       
#################################################
# Plot results

# Noise parameters
# Standard deviation (spread or “width”) of the distribution. Must be non-negative
magnitude = 1 # normalized data base 

# Define magnitude
magnitude = np.linspace(0, magnitude, 11)

datasets = [
# # # 'servo', 
'servo_norm', 
# # # 'car-evaluation', 
#'car-evaluation_norm', 
# # 'breast-tissue', 
#'breast-tissue_norm',
# # # 'Engine1', 
#'Engine1_norm', 
# # # 'xd6', 
#'xd6_norm', 
# # # 'hayes-roth',
# 'hayes-roth_norm',
# # # 'rabe_131',
#'rabe_131_norm',
# # # 'visualizing_environmental',
#'visualizing_environmental_norm',
# # # 'diggle_table_a2',
# 'diggle_table_a2_norm',
# # # 'newton_hema',  
#'newton_hema_norm',  
# # # 'wisconsin',
#'wisconsin_norm',
# # # 'fri_c4_250_100',
#'fri_c4_250_100_norm',
# # # 'conference_attendance',
# 'conference_attendance_norm',
# # # 'tic-tac-toe',
# 'tic-tac-toe_norm',
# # # 'qsar-biodeg',
# 'qsar-biodeg_norm',
# # # 'cmc',
# 'cmc_norm',
# # # 'heart-statlog',
# 'heart-statlog_norm'
]

datasets_2 = [
'cnae-9',                    
'AP_Breast_Kidney',    
'AP_Endometrium_Breast',        
'AP_Ovary_Lung',               
'OVA_Uterus',              
'micro-mass',                  
'har',                        
'eating',                      
'oh5.wc',                        
'leukemia']                        


metrics = ['Rand Index', 'Calinski-Harabasz Score', 'Fowlkes-Mallow Index']
methods = ['KISOMAP', 'ISOMAP', 'UMAP']

cols = 3
rows = 3
pages = 1 if len(datasets)//5 == 0 else len(datasets)//5
idx_db = 0


for p in range(1, pages+1):
    fig, axs = plt.subplots(rows, cols, figsize=(10, 5))
    
    for i, dataset in enumerate(datasets[idx_db:p*rows]):
        if i < rows:
            for j, metric in enumerate(metrics):
                ax = axs[i, j]  
                for method in methods:
                    ax.plot(magnitude, results[dataset][method][j], label=method)
                    
                if j == 0:
                    ax.set_ylabel(dataset.replace('_norm', ''))  # Set the y label here
                    plt.setp(ax.get_yticklabels(), visible=True)
                else:
                    plt.setp(ax.get_yticklabels(), visible=True) 
                if i == 0:
                    ax.set_title(metric)  
                if i == 0 and j==2:
                    ax.legend()
    
    idx_db += rows
    plt.savefig('first_dataset_results_page_' + str(p) +'.jpeg',dpi=300,format='jpeg')
    plt.show()