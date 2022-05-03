from helper_funcs_ConnectomeCompare import get_Caronlike, alignConnectomes, shufmat, confidence_interval, shufmat_indegree_only, run_PCA, jsanalysis, plot_ACP_updated

from scipy.spatial import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from dictances import bhattacharyya
from sklearn.decomposition import PCA
from matplotlib.ticker import MaxNLocator

W_Tatsuya_Orco, Tatsuya_Orco = get_Caronlike('Tatsuya_D_Mel_Orca_VC3Sep.csv')
W_Tatsuya_Control, Tatsuya_Control = get_Caronlike('Tatsuya_D_Mel_Control_VC3Sep.csv')

def PCA_Tatsuya(W1, W2, W1_label, W2_label):  # pcs of W1 compared to pcs of W2 fixed with W_Tatsuya_Control biases
    PC_components = 20
    n = 1000 # Number of repeats for random model results
    title = W1_label+'vs'+W2_label

    # Run PCA on W1 data
    W = np.copy(W1.T)
    pca = PCA(n_components = PC_components) 
    covar_matrix = pca.fit(W)
    variances = covar_matrix.explained_variance_ratio_ # Var ratios

    variances_rand_list = []
    # Run random models
    for i in range(n):
        W_rand = shufmat(W2, W_Tatsuya_Control)
        pca_rand = PCA(n_components = PC_components)
        covar_matrix_rand = pca_rand.fit(W_rand)
        variances_rand = covar_matrix_rand.explained_variance_ratio_ # Var ratios
        variances_rand_list.append(variances_rand)

    variances_rand_list = np.array(variances_rand_list) # (n, PC_components)
    variances_mean = np.mean(variances_rand_list, axis=0)

    lowers = []
    highers = []
    for i in range(PC_components):
        lower, higher = confidence_interval(variances_rand_list[:, i])
        lowers.append(lower)
        highers.append(higher)
    errors = np.array([variances_mean-lowers, highers-variances_mean])

    df_TO = pd.DataFrame(data={"component":range(PC_components), "variance explained "+W1_label:variances, "variance explained "+W2_label: variances_mean, "confidence interval lower":errors[0], "confidence interval higher":errors[1]})

    df_TO.to_csv("pca_"+title+".csv")

    # Plotting
    fig = plt.figure(figsize=(5,3))
    ax1 = fig.add_subplot(111)

    ax1.scatter(range(PC_components), variances, label=W1_label)
    ax1.scatter(range(PC_components), variances_mean, label=W2_label)
    ax1.errorbar(range(PC_components), variances_mean, errors, fmt='none')
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.ylabel('Fraction Variance Explained')
    plt.xlabel('Component')
    plt.title(title)


    plt.legend()
    plt.savefig(title+".png")
    plt.show()


    # Extract projections
    projected = pd.DataFrame(pca.fit_transform(W1)) 
    projected_rand = pd.DataFrame(pca_rand.fit_transform(W_rand))

PCA_Tatsuya(W_Tatsuya_Control, W_Tatsuya_Control, 'Orco+', 'Orco+_fixed')

PCA_Tatsuya(W_Tatsuya_Orco, W_Tatsuya_Orco, 'Orco-', 'Orco-_fixed')

PCA_Tatsuya(W_Tatsuya_Orco, W_Tatsuya_Control, 'Orco-', 'Orco+_fixed')

PCA_Tatsuya(W_Tatsuya_Control, W_Tatsuya_Orco, 'Orco+', 'Orco-_fixed')