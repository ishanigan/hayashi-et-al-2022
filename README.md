[The link to the raw data will be pasted here once the data upload is complete.]

# hayashi et al. 2022 analysis

Analysis code for Hayashi et al.

# Figure descriptions

*volcano.pdf/volcano_tatsuya.pdf:* This shows all comparisons between pairs of glomeruli across species, with the magnitude of the effect on the x-axis and the p-value on the y-axis.

*volcano_[1-4].pdf:* Same as the above, but for each individual comparison across males/females or species

*Connection_probabilities.pdf:* Plots the average connection probability for each glomerulus (fraction of connections from a specific glomerulus compared to the total number of connections) for different datasets. Each figure in this set has a short description and conclusion.

*Conditional_input_matrices.pdf:* These conditional input matrices are computed using the method detailed in Zheng et al. 2020. In the raw conditional input matrix, each cell in the matrix shows the probability that given input from the row PN type, the KC will also get input from the column PN type. This obtained conditional probability is then compared to the distribution of counts generated using a null model to obtain the final conditional input matrix. Each cell then represents the deviation of this conditional probability from the null models used. The community glomeruli are grouped to the top left so that any community structure can be easily seen. Each figure in this set has a short 
conclusion.

*JSDists_Ellis.pdf/JSDists_Tatsuya.pdf:* Shannon-Jensen distances between the glomerulus connection probability distributions for each pair of datasets, as well as distances between the conditional input matrices for each pair of datasets. More details for each figure are in the file.
   
