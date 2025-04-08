#VAE
embed_dim = 32
########################################################################
#SVM
BEST_PARAMS = {'kernel': 'poly',
 'C': 421.6699936797758,
 'gamma': 0.00021129844913526583,
 'degree': 2,
 'coef0': 0.6569561965570723,
 'class_weight': None}
BEST_PARAMS.update({'probability': True, 'random_state': 42})
############################################################
n_splits = 2  # for optuna SVM you can increase splits for more robust estimates
#########################################################################
#SHAP
IMP_FEATURES = ['embed_nuc_11',
 'embed_cell_3',
 'embed_cell_28',
 'embed_nuc_25',
 'embed_cell_16',
 'embed_nuc_3',
 'embed_nuc_21',]
##############################################################################
other_features = ['rel_area','cell_area','distance','cell-nuc_longest_angle']
cell_features = [f"embed_cell_{i}" for i in range(1, embed_dim+1)]
nuc_features = [f"embed_nuc_{i}" for i in range(1, embed_dim+1)]