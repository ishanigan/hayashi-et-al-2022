# Imports
import numpy as np
import pandas as pd   
import pickle
from matplotlib import pyplot as plt
import scipy.stats as st
from scipy import linalg
import math
import warnings 
warnings.filterwarnings('ignore')
from sklearn.decomposition import PCA
#import rcca
from mpl_toolkits.mplot3d import Axes3D 
import random
from scipy.spatial import *
import seaborn as sns

# Global Variables
PC_components = 21
datapath="data/"
glom_types = pd.read_csv('data/glomtypes.csv')


## Dataset extraction/alignment ##

# Internal helper function for obtaining Neuprint data
   
#returns 8 things:
#Win, the matrix of incoming connections (post=rows, pre=columns)
#Wout, the matrix of outgoing connections (post=rows, pre=columns)
#c, dataframe containing information about the neurons (same order as rows/columns of Win/Wout)
#cin, dataframe containing information about the upstream neurons (same order as columns of Win)
#cout, dataframe containing information about the downstream neurons (same order as rows of Wout)
#conns, a list with conns[i] containing the information about all connections of neuron i
#syns_out, a list with syns_out[i] containing the information about outgoing connections of neuron i
#syns_in, a list with syns_in[i] containing the information about incoming connections of neuron i
def genW2(filestr):
    print("loading from file")
    c,conns,syns_out,syns_in,out_syntots = pickle.load(open(filestr+".pkl","rb"))
    N = len(c) #number of neurons

    print("processing connections")
    #add information about the matching neuron to each connection dataframe
    for ii in range(N):
        #rename the column containing synapse counts to 'syncount'
        conns[ii].rename(columns={conns[ii].columns[3]: 'syncount'},inplace=True)
        #add data about the neuron that this connection involves
        conns[ii]["neuronbodyId"] = c.bodyId[ii] 
        conns[ii]["neuronname"] = c.name[ii] 

    conns_flat = pd.concat(conns,sort=False)
    conns_flat.loc[conns_flat.name == 0,"name"] = ""

    conns_in = conns_flat[conns_flat.relation == "upstream"].reset_index(drop=True)
    conns_out = conns_flat[conns_flat.relation == "downstream"].reset_index(drop=True)

    M_in = len(conns_in.bodyId.unique()) #number of unique neurons upstream of KCs
    M_out= len(conns_out.bodyId.unique()) #number of unique neurons downstream of KCs

    print("generating weight matrices")
    conns_in_unique = conns_in.drop_duplicates("bodyId")
    conns_out_unique = conns_out.drop_duplicates("bodyId")
    conns_in_unique = conns_in_unique.sort_values(by="name")
    conns_out_unique = conns_out_unique.sort_values(by="name")
    conns_in_unique.reset_index(drop=True,inplace=True)
    conns_out_unique.reset_index(drop=True,inplace=True)

    #generate hash functions that convert bodyIds into the index that will be used for the weight matrix
    neuron_hash = dict(zip(c.bodyId,np.arange(N))) #goes from neuron bodyId to index
    in_hash = dict(zip(conns_in_unique.bodyId,np.arange(M_in))) #goes from upstream bodyId to index
    out_hash = dict(zip(conns_out_unique.bodyId,np.arange(M_out))) #goes from downstream bodyId to index

    #make Win
    postinds = np.vectorize(neuron_hash.get)(conns_in.neuronbodyId)
    preinds = np.vectorize(in_hash.get)(conns_in.bodyId)

    Win = np.zeros([N,M_in])
    Win[postinds,preinds] = conns_in.syncount

    #make Wout
    postinds = np.vectorize(out_hash.get)(conns_out.bodyId)
    preinds = np.vectorize(neuron_hash.get)(conns_out.neuronbodyId)

    Wout = np.zeros([M_out,N])
    Wout[postinds,preinds] = conns_out.syncount

    cin = conns_in_unique[["bodyId","name"]]
    cout = conns_out_unique[["bodyId","name"]]
    
    #add postsynaptic total to downstream neurons
    #out_syntots_flat = pd.concat(out_syntots,sort=False)
    #out_syntots_flat = out_syntots_flat.drop_duplicates()
    #postinds = np.vectorize(out_hash.get)(out_syntots_flat.bodyId)

    #cout["tot"] = None
    #cout.tot[postinds] = out_syntots_flat.post.values.astype(int)

    print("done")

    return Win,Wout,c,cin,cout,conns,syns_out,syns_in

# Get Neuprint weights and glomerulus descripters as a dataframe
def get_Neuprint():
	Win,Wout,c,cin,cout,conns,syns_out,syns_in = genW2("data/kcs_nosyn")
	N = len(c)

	### annotate known inputs to KCs based on Greg's data and data about visual inputs ###

	#Greg's data
	#read info about uPN bodyIds and the glomeruli they belong to
	upns = pd.read_csv(datapath+"jefferis_glomerular_data_final/FIB_uPNs.csv", sep=',')
	validinds = upns.skid.isin(cin.bodyId)
	print("found",np.sum(validinds),"of",len(upns),"uPN IDs upstream of KCs")
	# print("\nPNs in Greg's data not found in EM:")
	# print(*upns[~validinds].skid.values,sep='\n')

	upns = upns[validinds] #only keep uPNs that exist in the list of neurons upstream of KCs

	#do same for mPNs and VP-PNs
	mpns = pd.read_csv(datapath+"jefferis_glomerular_data_final/FIB_mPNs.csv", sep=',')
	vppns = pd.read_csv(datapath+"jefferis_glomerular_data_final/FIB_VP_PNs.csv", sep=',')
	#vppns.val = np.array(["thermo_hygro" for i in range(len(vppns))])
	#read info about the valence of each glomerulus, add this to the above data
	gloms_val = pd.read_csv(datapath+"jefferis_glomerular_data_final/uPNsdf_FIB_FAFB.csv", sep=',')
	Nglom = len(gloms_val)

	upns["val"] = "" #add a column to the list of uPNs that contains their valence
	upns["priority"] = 1 #this will be used to sort which come first when saving the data

	# print(*upns[~validinds].val.values,sep='\n')



	allpns = pd.concat([upns,mpns,vppns],sort=False,ignore_index=True)

	#visual neuron data
	visual = pd.read_csv(datapath+"visual_inputs_Jack.csv")#pd.read_excel(datapath+"li_visual/v3.xlsx")
	#visual.rename(columns={"Body ID": "bodyId"},inplace=True)
	visual["priority"] = 4

	#generate subset of weight matrix and cin corresponding to inputs with valence label
	all_val = pd.concat([visual.bodyId,allpns.skid]) #all bodyIds that have a valence label
	validinds = cin.bodyId.isin(all_val)

	#find PNs that are in dataset but not in Greg's data
	allpninds = cin.name.str.contains("PN")
	unmatchedpns = cin[allpninds & (~validinds)]
	# print("\nPNs not matched to Greg's data:")
	# print(*unmatchedpns.name.values,sep='\n')

	Win_val = Win[:,validinds]
	cin_val = cin[validinds].copy()
	cin_val.reset_index(drop=True,inplace=True)

	cin_val["glom"] = ""
	cin_val["val"] = ""
	cin_val["pos_neg"] = ""
	cin_val["Type"] = ""
	cin_val["priority"] = ""

	#make everything lower case
	allpns.val = allpns.val.str.lower()

	#annotate cin_val with the valence, glomerulus, and type 
	for ii in allpns.index:
	    cin_val.loc[cin_val.bodyId == allpns.skid[ii],"glom"] = allpns.glomerulus[ii]
	    cin_val.loc[cin_val.bodyId == allpns.skid[ii],"Type"] = allpns.type[ii]
	    cin_val.loc[cin_val.bodyId == allpns.skid[ii],"priority"] = allpns.priority[ii]
	for ii in visual.index:
	    cin_val.loc[cin_val.bodyId == visual.bodyId[ii],"val"] = "visual"
	    cin_val.loc[cin_val.bodyId == visual.bodyId[ii],"Type"] = "visual"
	    cin_val.loc[cin_val.bodyId == visual.bodyId[ii],"priority"] = visual.priority[ii]


	#reorder by valence
	cin_val.sort_values(by=["priority","val"],inplace=True)
	Win_val = Win_val[:,cin_val.index]
	cin_val.reset_index(drop=True,inplace=True)


	valences =  pd.read_csv(datapath+"jefferis_glomerular_data_final/odour_scenes.csv", sep=',')


	for i in valences.index:
	    glom = valences.glomeruli[i]
	    for pn in range(len(cin_val.name)):
	        if glom == cin_val.glom[pn]:
	            cin_val.val[pn] = valences.odour_scene[i]
	            cin_val.pos_neg[pn] = valences.valence[i]



	###

	# for val in cin_val.val.unique():
	#     if pd.isna(val):
	#         print("mean W for",val,":",np.mean(Win_val[:,cin_val.val.isna()]))
	#     else:
	#         print("mean W for",val,":",np.mean(Win_val[:,cin_val.val==val]))

	Win_vis = Win_val[:,cin_val.val == "visual"]
	frac_vis = np.sum(Win_vis,1)/np.sum(Win_val,1) #fraction of synaptic input from visual neurons
	frac_vis[np.isnan(frac_vis)] = 0

	in_deg = np.sum(Win_val > 0,1)

	in_deg_vis = in_deg[frac_vis > 0.8]
	in_deg_nonvis = in_deg[frac_vis < 0.2]

	total_in_deg = np.array([np.sum(x.relation == "upstream") for x in conns])
	total_in_deg_vis = total_in_deg[frac_vis > 0.8]
	total_in_deg_nonvis = total_in_deg[frac_vis < 0.2]

	total_in_syns = np.array([np.sum((x.relation == "upstream")*x.syncount) for x in conns])
	total_in_syns_vis = total_in_syns[frac_vis > 0.8]
	total_in_syns_nonvis = total_in_syns[frac_vis < 0.2]


	vals = cin_val.val.astype(str).values
	valswitch = np.append(0,1+np.where(vals[0:-1] != vals[1:])[0]) #indices of valence boundaries
	valnames = vals[valswitch]
	for ii in range(len(valnames)):
	    if len(valnames[ii]) > 32:
	        valnames[ii] = valnames[ii][:32]

	W = pd.DataFrame(np.copy(Win_val))

	thermo_hygro_bodyIds = [1639243580,
	1639234609,
	1727979406,
	5901222731,
	2065197353,
	2038617453,
	2069644133,
	1225100939,
	1788676257,
	5813069447,
	5813056072,
	5813063239,
	1974846717,
	1881401277,
	1943811736,
	1881059780,
	1943812176,
	1912777226,
	5812996748,
	1858901026,
	850717220,
	509626405,
	850708783,
	603785283,
	1975187675,
	1975187554,
	1975878958,
	5813040515,
	2069648663,
	1944502935,
	1755556097,
	1039237469,
	543010474,
	634759240,
	663432544,
	1789013008,
	663787020,
	664814903,
	5813077788,
	2100010400,
	1883443112]

	for i in range(len(cin_val)):
	    if cin_val.bodyId[i] in thermo_hygro_bodyIds:
	       cin_val.val[i]="thermo_hygro"

	return W, cin_val, c


# Any of the Caron datasets
def get_Caronlike(file):
	Caron = pd.read_csv('data/'+file)
	headers = list(Caron.columns.values)

	W_Caron = Caron.iloc[:, 5:]
	headers = list(W_Caron.columns.values)
	W_Caron = W_Caron.fillna(0.0)
	W_Caron = W_Caron.sort_index(axis=1)
	return W_Caron, Caron

# Return binary FAFB weight matrix as a dataframe
def get_FAFB():
	bin_mat = np.load('data/Zheng_pn_kc_bi_conn.npy') # Shape (1356, 113)
	pns = [2863104, 57349, 57353, 11544074, 16, 23569, 57361, 43539, 57365, 11524119, 192547, 57381, 36390, 57385, 23597, 24622, 37935, 400943, 775731, 67637, 11544121, 57402, 22594, 57410, 57414, 30791, 57418, 57422, 68697, 1775706, 23134, 56424, 41578, 27246, 33903, 22132, 35447, 37513, 32399, 24726, 57499, 27295, 771242, 30891, 57003, 57516, 51886, 45242, 24251, 40637, 49865, 1785034, 28876, 186573, 46800, 73937, 22744, 30434, 39139, 65762, 27884, 39668, 39682, 22277, 36108, 23829, 61221, 40749, 55085, 56623, 54072, 45882, 58686, 61773, 755022, 67408, 55125, 39254, 41308, 40306, 22906, 53631, 60799, 37250, 23432, 51080, 52106, 22422, 57241, 46493, 57246, 581536, 53671, 27048, 35246, 42927, 42421, 165303, 65465, 22976, 32214, 23512, 27611, 57307, 57311, 62434, 38885, 57319, 57323, 21999, 57333, 57337, 57341]
	types = pd.read_csv('data/Zheng_PN_subtypes')

	# Map each skid to the appropriate glom
	pns_gloms = []
	for id in pns:
		if id in types['skids'].values:
			pns_gloms.append(types[types['skids']==id]['gloms'].item())

	FAFB = pd.DataFrame(bin_mat)
	FAFB.columns = pns_gloms
	FAFB = FAFB.groupby(lambda x:x, axis=1).sum() # Group like glomeruli

	return FAFB

# Align the Caron and Neuprint data and return the aligned matrices
def alignConnectomes():
	# Obtain Neuprint and Caron weight matrices
	Win_val, cin_val, c =get_Neuprint()
	W_FAFB = get_FAFB()

	# VC3L+VC3M
	Win_val[285] = Win_val[74] + Win_val[75] + Win_val[76] + Win_val[77] + Win_val[78] + Win_val[79] 
	cin_val = cin_val.append({'glom': 'VC3L+VC3M'}, ignore_index=True)
	W_FAFB['VC3L+VC3M'] = W_FAFB['VC3m']+W_FAFB['VC3l']

	# Find list of gloms to delete
	delete_gloms = []
	neu_only = ['','VC3l', 'VC3m', 'VC3l?', 'VC5', 'VP1d + VP4', 'VP1d+', 'VP1l + VP3', 'VP1m + VP2', 'VP1m + VP5', 'VP1m+', 'VP2', 'VP2+', 'VP3', 'VP3 + VP1l', 'VP4', 'VP5+ and SEZ']
	for i in range(len(cin_val["glom"])):
	    if pd.isnull(cin_val["glom"][i]) == True:
	        delete_gloms.append(i)
	    if cin_val["glom"][i] in neu_only:
	        delete_gloms.append(i)

	# Delete gloms
	Win_val = Win_val.drop(columns=delete_gloms)
	cin_val = cin_val.drop(delete_gloms)
	W_FAFB = W_FAFB.drop(['VC3l', 'VC3m', 'VP3+VP11', 'VC5', 'VP2'], axis=1)

	# Re-threshold FAFB dataset to binarize it
	W_FAFB_thr = pd.DataFrame(np.where(W_FAFB>0, 1, 0))
	W_FAFB_thr.columns=W_FAFB.columns

	# Fix dataframe naming convention
	Win_val.columns=cin_val["glom"]
	W_Neuprint = Win_val.groupby(lambda x:x, axis=1).sum()

	# Create thresholded Neuprint dataset
	W_Neuprint_thr = pd.DataFrame(np.where(W_Neuprint>5, 1, 0))
	W_Neuprint_thr.columns=W_Neuprint.columns

	# Sort all dataframes so that gloms are alphabetically ordered
	W_Neuprint = W_Neuprint.sort_index(axis=1)
	W_Neuprint_thr = W_Neuprint_thr.sort_index(axis=1)
	W_FAFB_thr = W_FAFB_thr.sort_index(axis=1)

	# Grab the Caron matrices
	W_OGCaron, OGCaron = get_Caronlike('OriginalCaron2020_DMelanFemale.csv')
	W_Caron2013, Caron2013 = get_Caronlike('caron2013.csv')
	W_Mel_Male, Ellis_Mel_M = get_Caronlike('Ellis_D_Melanogaster_Male.csv')
	W_Mel_Female, Ellis_Mel_F = get_Caronlike('Ellis_D_Melanogaster_Female.csv')
	W_Sec_Female, Ellis_Sec_F = get_Caronlike('Ellis_D_Sec_Female.csv')
	W_Sim_Female, Ellis_Sim_F = get_Caronlike('Ellis_D_Sim_Female.csv')

	# Sanity check two matrices
	caron = list(W_OGCaron.columns.values)
	neuprint = list(W_Neuprint.columns.values)
	fafb = list(W_FAFB_thr.columns.values)

	# Neu
	print("there are " + str(len(neuprint)) + " neuprint glomeruli")
	print(neuprint)

	# Caron 
	print("there are " + str(len(caron)) + " caron glomeruli")
	print(caron)

	# FAFB 
	print("there are " + str(len(fafb)) + " fafb glomeruli")
	print(fafb)

	# Intersection
	intersect = list(common_member(neuprint, fafb))
	print("there are " + str(len(intersect))+ " common glomeruli")
	print(intersect)

	# Only Neuprint
	onlyneu = np.setdiff1d(neuprint, intersect)
	print("there are " + str(len(onlyneu))+ " only neuprint glomeruli")
	print(onlyneu)

	# Only FAFB
	onlyfafb = np.setdiff1d(fafb, intersect)
	print("there are " + str(len(onlyfafb))+ " only fafb glomeruli")
	print(onlyfafb)



	return W_Neuprint, W_Neuprint_thr, W_OGCaron, W_Caron2013, W_Mel_Male, W_Mel_Female, W_Sec_Female, W_Sim_Female, W_FAFB_thr, cin_val, c, OGCaron  # Returns Neuprint, Neuprint thresholded, and Caron thresholded already

# Takes two lists and eturns a list of the common members 
def common_member(list_a, list_b):     
    a_set = set(list_a) 
    b_set = set(list_b) 
      
    # check length  
    if len(a_set.intersection(b_set)) > 0: 
        return(a_set.intersection(b_set))   
    else: 
        return("no common elements") 

# Shuffle matrix returning shuffled matrix mat_W with indegree of mat_X, keeping connection probs consistent
def shufmat(mat_W, mat_X):
    M = mat_W.shape[0]
    N = mat_W.shape[1]

    indeg = np.sum(mat_X>0,1)
    
    cprobs = np.mean(mat_W>0,0)
    cprobs = cprobs / np.sum(cprobs)

    Wshuf = np.zeros([M,N])

    for mi in range(M):
        num_inputs = np.random.choice(indeg)
        inds = np.random.choice(N,num_inputs,p=cprobs, replace=False)
        Wshuf[mi,inds] = 1

    return Wshuf

# Shuffle matrix returning shuffled matrix mat_W with indegree of mat_X
def shufmat_indegree_only(mat_W, mat_X):
    M = mat_W.shape[0]
    N = mat_W.shape[1]

    indeg = np.sum(mat_X>0,1)

    Wshuf = np.zeros([M,N])

    for mi in range(M):
        num_inputs = np.random.choice(indeg)
        inds = np.random.choice(N,num_inputs,replace=False)
        Wshuf[mi,inds] = 1

    return Wshuf

# Shuffle matrix keeping the in-degree and average connection probability consistent and return shuffled matrix
def shufmat_ACP_only(mat_W):
    M = mat_W.shape[0]
    N = mat_W.shape[1]

    indeg = np.sum(mat_W>0,1)
    mindeg = np.min(indeg)
    maxdeg = np.max(indeg)

    cprobs = np.mean(mat_W>0,0)
    cprobs = cprobs / np.sum(cprobs)

    Wshuf = np.zeros([M,N])

    for mi in range(M):
    	n = random.randint(mindeg, maxdeg)
    	inds = np.random.choice(N,n,p=cprobs,replace=False)
    	Wshuf[mi,inds] = 1

    return Wshuf

# Runs PCA on input, as well as null model matrices (with degree dist. null model) and plots fraction of variance explained
def run_PCA(W_in, W_label, shuffle_type, title):
	n = 1000 # Number of repeats for random model results

	# Run PCA on data
	W = np.copy(W_in.T)
	pca = PCA(n_components = PC_components) # features from figure 13A
	covar_matrix = pca.fit(W)
	variances = covar_matrix.explained_variance_ratio_ # Var ratios

	variances_rand_list = []
	# Run random models
	for i in range(n):
		if shuffle_type == 'original':
			W_rand = shufmat(W)
		elif shuffle_type == 'ACPonly':
			W_rand = shufmat_ACP_only(W)
		elif shuffle_type == 'Indegreeonly':
			W_rand = shufmat_indegree_only(W)
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
	    
	# Plotting
	fig = plt.figure(figsize=(5,3))
	ax1 = fig.add_subplot(111)

	ax1.scatter(range(PC_components), variances, label=W_label)
	ax1.scatter(range(PC_components), variances_mean, label='Shuffle')
	ax1.errorbar(range(PC_components), variances_mean, errors, fmt='none')

	plt.ylabel('Fraction Variance Explained')
	plt.xlabel('Component')
	plt.title(title)

	plt.legend()
	plt.show()

	# Extract projections
	projected = pd.DataFrame(pca.fit_transform(W)) 
	projected_rand = pd.DataFrame(pca_rand.fit_transform(W_rand))

	return projected, projected_rand

# Return canonical correlations/variances and cca object from two matrices
def run_CCA(mat1, mat2):
	nComponents = PC_components # min(p,q) components
	cca = rcca.CCA(kernelcca = False, reg = 0., numCC = nComponents, verbose=False)
    # train on data
	cca.train([mat1, mat2])

	cancorrs = cca.cancorrs
        
	return cancorrs, cca


def plot_ACP(list_weights, list_labels, title):
	# Create figure
	acp_fig = plt.figure(figsize=(15,10))
	ax1 = acp_fig.add_subplot(111)

	acps = [] # list of all acps
	acp_data=[] # data to csv

	for i in range(len(list_weights)):
		W = list_weights[i]
		acp = np.mean(W>0,0)
		acp = acp / np.sum(acp)
		acps.append(acp)

	acpsum = sum(acps).values

	indsort = np.argsort(acpsum)[::-1]
        
	for i in range(len(list_weights)):
		acp = acps[i]
		ax1.plot(acp.values[indsort], marker='.', label=list_labels[i])
		acp_data.append(acp.values[indsort])

		
	plt.xticks(np.arange(len(acp.values)),acp.index.values[indsort],rotation=90, fontsize=15)
	plt.ylabel('average connection probability')
	plt.title(title)
	plt.legend()

	plt.show()

	# Generate CSV
	output = pd.DataFrame(data=np.array(acp_data).T, columns=list_labels)
	output.index=acp.index.values[indsort]
	output.to_csv(title+'.csv')


	return acps

def compute_2propPvalues(list_weights):
	W1 = list_weights[0]
	W2 = list_weights[1]

	local_W1 = W1.sum()
	local_W2 = W2.sum()
	total_W1 = local_W1.sum()
	total_W2 = local_W2.sum()
	acp_W1 = local_W1 / total_W1
	acp_W2 = local_W2 / total_W2
	pi = (local_W1 + local_W2) / (total_W1 + total_W2)
	z = (acp_W1 - acp_W2) / np.sqrt(pi*(1-pi)*((1/total_W1)+(1/total_W2)))
	pvals = st.norm.sf(abs(z))*2

	# Configure p values database
	df = pd.DataFrame()
	df['Glom'] = list(W1.columns.values)
	df['PVal'] = pvals
	df['Significant Diff?'] = pvals < 0.05
	df = df.sort_values(ascending=True, by='Glom')

	return df 

def plot_PCARepresentation(projections, title):
	plt.figure()
	plt.scatter(projections[0], projections[1], color=glom_types['color'])
	plt.xlabel('component 1')
	plt.ylabel('component 2')
	plt.title(title)
	plt.show()

# Takes in two lists of PC matrs and plots the subspace angle between pairs of weight matrices, for different numbers of PCs
def subspace_angles(list_PCs_1, list_PCs_2, list_labels, title):
	num_pairs = len(list_PCs_1)

	plt.figure(figsize=(20,10))

	# For each pair of PC matrices
	for pair in range(num_pairs):
		sub_angles = []
		for i in range(1,PC_components):
		    seg1 = list_PCs_1[pair].loc[:, :i]
		    seg2 = list_PCs_2[pair].loc[:, :i]
		    
		    sub_angles.append(np.min(linalg.subspace_angles(seg1, seg2)))
		    
		plt.plot(sub_angles, label=list_labels[pair])

	plt.legend()
	plt.xlabel('PC Components')
	plt.ylabel('Smallest subspace angle')
	plt.title(title)

# Returns Caron and Neuprint weight matrices separated by kc subtype
def kc_subtypes():
	W_Neuprint, W_Neuprint_thr, W_Caron, cin_val, c, Caron, FAFB = alignConnectomes()

	# Find indices of KC subtypes in KC list
	inds_gamma_N = c[(c['instance']=='KCg') | (c['instance']=='KCg-d') | (c['instance']=='KCg(super)')].index.values
	# From 1328 to 2028 inclusive are gammas

	inds_ab_N = c[(c['instance']=='KCab-a') | (c['instance']=='KCab-c') | (c['instance']=='KCab-p') | (c['instance']=='KCab-s')].index.values
	# From 438 to 1325 inclusive are abs

	inds_a_b_N = c[(c['instance']=='KC\'B\'_L') | (c['instance']=='KCa\'b\'')].index.values
	# From 102 to 437 inclusive are a_b_s

	inds_gamma_C = Caron[(Caron['KC Type']=='gamma')].index.values
	inds_ab_C = Caron[(Caron['KC Type']=='alpha/beta')].index.values
	inds_a_b_C = Caron[(Caron['KC Type']=='alpha\'/beta\'')].index.values

	# Splice gamma from Neuprint dataset
	W_Gamma_N = W_Neuprint_thr.iloc[1328:2029, :]
	W_AB_N = W_Neuprint_thr.iloc[438:1326, :]
	W_A_B_N = W_Neuprint_thr.iloc[102:438, :]
	W_Gamma_C = W_Caron.iloc[inds_gamma_C, :]
	W_AB_C = W_Caron.iloc[inds_ab_C, :]
	W_A_B_C = W_Caron.iloc[inds_a_b_C, :]

	return W_Gamma_N, W_AB_N, W_A_B_N, W_Gamma_C, W_AB_C, W_A_B_C

def plot_ACP_community(list_weights, list_labels, title):

    community = ['VM3', 'DL2d', 'VA2', 'VA4', 'DP1m', 'DM1', 'DM2', 'VM2', 'DM3', 'DM4', 'D', 'DA1', 'DA2', 'DA3', 'DA4l', 'DA4m', 'DC1', 'DC2', 'DC3', 'DC4',
       'DL1', 'DL2v', 'DL3', 'DL4', 'DL5',
       'DM5', 'DM6', 'DP1l', 'V', 'VA1d', 'VA1v', 'VA3', 
       'VA5', 'VA6', 'VA7l', 'VA7m', 'VC1', 'VC2', 'VC3L+VC3M', 'VC4', 'VL1',
       'VL2a', 'VL2p', 'VM1', 'VM4', 'VM5d', 'VM5v', 'VM7d',
       'VM7v', 'VP1m']
    
    # Create figure
    acp_fig = plt.figure(figsize=(15,10))
    ax1 = acp_fig.add_subplot(111)

    acps = [] # list of all acps

    for i in range(len(list_weights)):
        W = list_weights[i]
        acp = np.mean(W>0,0)
        acp = acp / np.sum(acp)
        acp = acp.reindex(community)

        # Only one weight matrix passed in
        if (len(list_weights) == 1):
            acp = acp.sort_values(ascending=False)
            y_avg = np.array([np.mean(acp)]*len(acp))
            ax1.plot(acp.index, y_avg, linestyle='dashed', color = 'grey', label='Mean')
        ax1.plot(acp, marker='.', label=list_labels[i])
        acps.append(acp)

    # Add color labels
    glom_colors = glom_types['color']
    my_colors = []
    for i in range(len(community)):
        glom = community[i]
        my_colors.append(glom_types[glom_types['glom']==glom]['color'].item())
    
    plt.xticks(rotation=90, fontsize=15)
    for ticklabel, tickcolor in zip(plt.gca().get_xticklabels(), my_colors):
        ticklabel.set_color(tickcolor)
        
    plt.xticks(rotation=90, fontsize=15)
    plt.ylabel('Probability')
    plt.title(title)
    plt.legend()

    plt.show()

def plot_ACP_noni(list_weights, list_labels, title):

    noni = ['DL2d', 'DM1', 'DM2', 'VA2', 'VC3L+VC3M', 'VM5d', 'D', 'DA1', 'DA2', 'DA3', 'DA4l', 'DA4m', 'DC1', 'DC2', 'DC3', 'DC4', 'DL1', 'DL2v', 'DL3', 'DL4', 'DL5',  'DM3', 'DM4', 'DM5', 'DM6', 'DP1l', 'DP1m', 'V', 'VA1d', 'VA1v', 'VA3', 'VA4', 'VA5', 'VA6', 'VA7l', 'VA7m', 'VC1', 'VC2', 'VC4', 'VL1', 'VL2a', 'VL2p', 'VM1', 'VM2', 'VM3', 'VM4', 'VM5v', 'VM7d', 'VM7v', 'VP1m']
    
    # Create figure
    acp_fig = plt.figure(figsize=(15,10))
    ax1 = acp_fig.add_subplot(111)

    acps = [] # list of all acps

    for i in range(len(list_weights)):
        W = list_weights[i]
        acp = np.mean(W>0,0)
        acp = acp / np.sum(acp)
        acp = acp.reindex(noni)

        # Only one weight matrix passed in
        if (len(list_weights) == 1):
            acp = acp.sort_values(ascending=False)
            y_avg = np.array([np.mean(acp)]*len(acp))
            ax1.plot(acp.index, y_avg, linestyle= 'dashed', color = 'grey', label='Mean')
        ax1.plot(acp, marker='.', label=list_labels[i])
        acps.append(acp)

    # Add color labels
    glom_colors = glom_types['color']
    my_colors = []
    for i in range(len(noni)):
        glom = noni[i]
        my_colors.append(glom_types[glom_types['glom']==glom]['color'].item())
    
    plt.xticks(rotation=90, fontsize=15)
    for ticklabel, tickcolor in zip(plt.gca().get_xticklabels(), my_colors):
        ticklabel.set_color(tickcolor)
        
    plt.xticks(rotation=90, fontsize=15)
    plt.ylabel('Probability')
    plt.title(title)
    plt.legend()

    plt.show()


# Helper function for running 'subspace_angles_pair'
def run_PCA_pair(W_in):
    n = 1000# Number of repeats for random model results

    # Run PCA on data
    W = np.copy(W_in)
    pca = PCA(n_components = PC_components) # features from figure 13A
    projected = pd.DataFrame(pca.fit_transform(W)) 
    
    # Random models

    rand_list = []
    # Run random models
    for i in range(n):
        W_rand = shufmat(W)
        pca_rand = PCA(n_components = PC_components)
        projected_rand = pca_rand.fit_transform(W_rand)
        rand_list.append(projected_rand)


    return projected, rand_list

# Subspace angle analysis for a pair of datasets, with confidence intervals
def subspace_angles_pair(W_1, W_2, title):
    projected1, projected_rand_1 = run_PCA_pair(W_1.T)
    projected2, projected_rand_2 = run_PCA_pair(W_2.T)

    plt.figure(figsize=(20,10))

    data_angles = []
    for i in range(1,PC_components):
        seg1 = projected1.loc[:, :i]
        seg2 = projected2.loc[:, :i]

        data_angles.append(np.min(linalg.subspace_angles(seg1, seg2)))

    plt.plot(list(range(0, PC_components-1)), data_angles, label='data')

    runs = []
    for j in range(1000):
        shuf_angles = []
        for i in range(1,PC_components):
            seg1 = pd.DataFrame(projected_rand_1[j]).loc[:, :i]
            seg2 = pd.DataFrame(projected_rand_2[j]).loc[:, :i]

            shuf_angles.append(np.min(linalg.subspace_angles(seg1, seg2)))
        runs.append(shuf_angles)
    runs = np.array(runs)

    lowers = []
    highers = []
    for i in range(PC_components-1):
    	lower, higher = confidence_interval(runs[:, i])
    	lowers.append(lower)
    	highers.append(higher)
	

    y = np.mean(runs, axis=0)
    plt.plot(y, label='shuffle')
    plt.fill_between(list(range(0, PC_components-1)), lowers, highers, alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')


    plt.legend()
    plt.xlabel('PC Components')
    plt.ylabel('Smallest subspace angle')
    plt.title(title)
    return runs

def confidence_interval(a):
    sorted = np.sort(a)
    boundary = int(np.around(0.026*len(a)))
    lower = sorted[boundary]
    higher = sorted[-boundary]
    return lower, higher

# Form random matrices with structure for sensitivity testing
def struct_mat(num_kcs, num_gloms, p):
    kcshuff = list(range(num_kcs))
    random.shuffle(kcshuff)
    glomshuff = list(range(num_gloms))
    random.shuffle(glomshuff)
    weights = np.zeros((num_kcs, num_gloms))
    for kc in range(num_kcs):
        num_claws = choices(indClaws, freqClaws)[0] # Pick number of claws for KC from Caron distribution
        s = np.random.uniform()
        if s > p:
            # Random KCs
            gloms = sample(range(num_gloms), num_claws) # Pick which gloms are connected
        else: 
            # Structured KCs
            if kc < 400:
                gloms = sample(glomshuff[0:10], num_claws)
            elif kc < 800:
                gloms = sample(glomshuff[10:20], num_claws)
            elif kc < 1200:
                gloms = sample(glomshuff[20:30], num_claws)
            elif kc < 1600:
                gloms = sample(glomshuff[30:40], num_claws)
            elif kc < 2000:
                gloms = sample(glomshuff[40:53], num_claws)
        for iglom in gloms:
            weights[kc, iglom] = 1
    plt.figure(figsize=(10,5))
    plt.imshow(weights.T, cmap='Greys_r', aspect='auto')
    return weights



