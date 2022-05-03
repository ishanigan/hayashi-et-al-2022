import pandas as pd
from scipy.stats import fisher_exact
import numpy as np
import matplotlib.pyplot as plt

from helper_funcs_ConnectomeCompare import get_Caronlike, plot_ACP, plot_ACP_updated, alignConnectomes, run_PCA, subspace_angles, shufmat, shufmat_indegree_only, shufmat_ACP_only, subspace_angles_pair, kc_subtypes 

W_Tatsuya_Orca, Tatsuya_Orca = get_Caronlike('Tatsuya_D_Mel_Orca_VC3Sep.csv')
W_Tatsuya_Control, Tatsuya_Control = get_Caronlike('Tatsuya_D_Mel_Control_VC3Sep.csv')
W_Tatsuya_Control_shuff = shufmat_indegree_only(W_Tatsuya_Orca, W_Tatsuya_Control)
W_Tatsuya_Control_shuff = pd.DataFrame(W_Tatsuya_Control_shuff, columns = W_Tatsuya_Control.columns)
W_Tatsuya_Control_shuff_strong = shufmat(W_Tatsuya_Control, W_Tatsuya_Control)
W_Tatsuya_Control_shuff_strong = pd.DataFrame(W_Tatsuya_Control_shuff_strong, columns = W_Tatsuya_Control.columns)

Nsheets = 4
Nglom = 51 #ignore "other" glomeruli
datasets = [W_Tatsuya_Control_shuff, W_Tatsuya_Control, W_Tatsuya_Orca, W_Tatsuya_Control_shuff_strong]
sheetnames = ["Orco +/+ shuffle","Orco +/+", "Orco -/-", "Orco +/+ fixed"]
gloms = W_Tatsuya_Control.columns

Ja = list() # List of raw binary connectivity matrices

for dataset in datasets:
    Ja.append(np.array(dataset))

#all pairwise comparisons

Npair = 6

pvs = np.zeros([Npair,Nglom])
ratio = np.zeros([Npair,Nglom])
comparisons = np.zeros([Npair,Nglom],dtype=object)
icount = 0
sia = np.zeros(Npair,dtype=int)
sja = np.zeros(Npair,dtype=int)

for si in range(Nsheets-1):
    print(si)
    Nkci = Ja[si].shape[0]
    for sj in range((si+1),Nsheets):
        #if si == 1: #skip Dmel males vs. Dsec females and Dsim females
         #   continue
        sia[icount] = si
        sja[icount] = sj

        Nkcj = Ja[sj].shape[0]

        for gi in range(Nglom):
            connsi = np.sum(Ja[si][:,gi])
            connsj = np.sum(Ja[sj][:,gi])
            table = np.array([[connsi, connsj],[Nkci-connsi,Nkcj-connsj]])
            out,pvs[icount,gi] = fisher_exact(table)
            if connsj > 0:
                ratio[icount,gi] = connsi/connsj

            comparisons[icount,gi] = sheetnames[si]+" vs. "+sheetnames[sj]+" "+str(gloms[gi])

        icount = icount+1

###
#determine significant p-values using Benjamini-Hochberg procedure to control false discovery rate

Ncomparisons = len(pvs.flatten())
slope = 0.1/Ncomparisons #10% false discovery rate

sortinds = np.argsort(pvs,axis=None)
pvsflat = pvs.flatten()
ratioflat = ratio.flatten()

pvsort = pvsflat[sortinds]
k = 1+np.arange(Ncomparisons)

diff = pvsort-(slope*k)

Nvalid = np.where(diff > 0)[0][0]


for ii in range(Nvalid):
    ind = np.unravel_index(sortinds[ii],pvs.shape)
    si = sia[ind[0]]
    sj = sja[ind[0]]
    glom = gloms[ind[1]]
    pv = pvs[ind]

    print(sheetnames[si],"vs.",sheetnames[sj],glom,"p-value",pv)

import pandas as pd
df = pd.DataFrame(data={"comparison":comparisons.flatten(), "ratio":ratioflat, "p-value": pvsflat, "Log fold change (base 2)":np.log2(ratioflat), "-Log p-value (base 10)":np.abs(np.log(pvsflat))})

df.to_csv("volcano.csv")

plt.scatter(np.log2(ratioflat),np.abs(np.log(pvsflat)),s=5)
plt.xlabel("Log fold change (base 2)")
plt.ylabel("-Log p-value (base 10)")
plt.tight_layout()
plt.show()
plt.savefig("volcano.pdf")

for icount in range(Npair):
    fig, ax = plt.subplots()
    plt.scatter(np.log2(ratio[icount,:]),np.abs(np.log(pvs[icount,:])),s=5)
    plt.xlabel("Log fold change (base 2)")
    plt.ylabel("-Log p-value (base 10)")
    plt.title(comparisons[icount,0][:-2])
    plt.tight_layout()
    fname = "volcano_"+str(icount+1)+".pdf"
    for i, txt in enumerate(gloms):
        ax.annotate(txt, (np.log2(ratio[icount,:])[i], np.abs(np.log(pvs[icount,:]))[i]), fontsize='x-small')
    plt.savefig(fname)
    plt.show()

