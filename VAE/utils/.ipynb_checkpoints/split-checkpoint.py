import numpy as np
from sklearn.model_selection import GroupShuffleSplit
import pandas as pd
def get_best_three_splits(X, y, groups, test_size=0.2, n_splits=100, random_state=42):
    """
    Generate up to three distinct (train_idx, test_idx) splits using GroupShuffleSplit
    that best preserve the overall positive-class ratio in both training and testing sets.
    Also prints group allocations, group ratios, sample ratios, and class ratios for 
    each chosen split.

    'Distinct' means no two chosen splits have the same set of test groups.

    Parameters
    ----------
    X : array-like
        Feature matrix or data points (used only for its length, not for group or label info).
    y : array-like
        Binary labels (0 or 1).
    groups : array-like
        Group labels for each sample (ensures that the same group is not split between
        training and testing).
    test_size : float, optional
        Proportion of data to be used for the test set, by default 0.2
    n_splits : int, optional
        Number of random candidate splits to consider, by default 100
    random_state : int, optional
        Random seed for reproducibility, by default 42
        
    Returns
    -------
    list of tuples
        A list of up to three tuples, each tuple being (train_idx, test_idx).
        The first tuple has the smallest class-ratio deviation, the second is the
        second-best, etc.
    """
    
    splitter = GroupShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
    overall_pos_ratio = np.mean(y == 1)
    
    # Collect all valid candidate splits and their deviation from the overall pos ratio
    candidate_splits = []
    
    for train_idx, test_idx in splitter.split(X, y, groups=groups):
        # Double-check no group overlap (should be guaranteed by GroupShuffleSplit)
        train_groups_set = set(groups[train_idx])
        test_groups_set = set(groups[test_idx])
        if len(train_groups_set.intersection(test_groups_set)) > 0:
            continue
        
        # Calculate positive-class ratio in train and test
        train_pos_ratio = np.mean(y[train_idx] == 1)
        test_pos_ratio = np.mean(y[test_idx] == 1)
        
        # Total deviation from overall ratio
        diff = abs(train_pos_ratio - overall_pos_ratio) + abs(test_pos_ratio - overall_pos_ratio)
        
        # Store (train_idx, test_idx, diff)
        candidate_splits.append((train_idx, test_idx, diff))
    
    if not candidate_splits:
        raise ValueError("No valid train/test splits found. Try increasing n_splits or revisiting data.")
    
    # Sort splits by deviation (ascending)
    candidate_splits.sort(key=lambda x: x[2])
    
    # Keep track of the top splits, ensuring test sets differ
    best_splits = []
    chosen_test_group_sets = []
    
    for (train_idx, test_idx, diff) in candidate_splits:
        test_groups_set = set(groups[test_idx])
        
        # Skip if this exact test-group set was already chosen
        if any(test_groups_set == already_chosen for already_chosen in chosen_test_group_sets):
            continue
        
        best_splits.append((train_idx, test_idx, diff))
        chosen_test_group_sets.append(test_groups_set)
        
        # Stop if we've collected 3
        if len(best_splits) == 3:
            break
    
    if not best_splits:
        raise ValueError("No non-duplicate test sets found among candidate splits.")
    
    # For printing ratios, get total groups and total samples
    total_groups = len(set(groups))
    total_samples = len(X)

    # Print out details for each chosen split
    for i, (train_idx, test_idx, diff) in enumerate(best_splits, start=1):
        train_groups_set = set(groups[train_idx])
        test_groups_set = set(groups[test_idx])
        
        train_groups_list = sorted(train_groups_set)
        test_groups_list = sorted(test_groups_set)
        
        # Class ratios
        train_pos_ratio = np.mean(y[train_idx] == 1)
        test_pos_ratio = np.mean(y[test_idx] == 1)
        
        # Group ratios (relative to total unique groups)
        train_group_ratio = len(train_groups_set) / total_groups
        test_group_ratio = len(test_groups_set) / total_groups
        
        # Sample ratios (relative to total samples)
        train_sample_ratio = len(train_idx) / total_samples
        test_sample_ratio = len(test_idx) / total_samples
        
        # print(f"\n=== Best Split #{i} ===")
        # print(f"Deviation from overall pos ratio: {diff:.4f}\n")
        
        # print(f"Train groups ({len(train_groups_set)}) ratio: {train_group_ratio:.2f} of total groups")
        # print(f"Test groups  ({len(test_groups_set)}) ratio: {test_group_ratio:.2f} of total groups")
        # print(f"Train groups list: {train_groups_list}")
        # print(f"Test groups  list: {test_groups_list}\n")
        
        # print(f"Train samples: {len(train_idx)} (ratio: {train_sample_ratio:.2f} of total)")
        # print(f"Test samples : {len(test_idx)} (ratio: {test_sample_ratio:.2f} of total)\n")
        
        # print(f"Train pos ratio: {train_pos_ratio:.3f}")
        # print(f"Test pos ratio : {test_pos_ratio:.3f}")
        # print(f"Overall pos ratio: {overall_pos_ratio:.3f}")
    
    # Return just (train_idx, test_idx) pairs
    return [(split[0], split[1]) for split in best_splits]



def prep_data(df,embeddings_cells ,embeddings_nucs,cells_np,nucs_np):
    df2 = df.copy()
    df2['cell-nuc_longest_angle']= np.abs(df2['cell-nuc_longest_angle'])
    df3 = df2.reset_index()
    other_features = ['rel_area','cell_area','distance','cell-nuc_longest_angle']
    cell_features = [f"embed_cell_{i}" for i in range(1, embeddings_cells.shape[1]+1)]
    nuc_features = [f"embed_nuc_{i}" for i in range(1, embeddings_nucs.shape[1]+1)]
    feature_names = other_features+ [f"embed_cell_{i}" for i in range(1, embeddings_cells.shape[1]+1)]+[f"embed_nuc_{i}" for i in range(1, embeddings_cells.shape[1]+1)]
    x1_features = other_features
    X1 = df2[x1_features].values
    X2 = np.array(embeddings_cells)
    X3 = np.array(embeddings_nucs)
    y = df2['ruptured_man'].astype(int).values 
    X = np.concatenate([X1,X2,X3], axis=1)
    best_splits =  get_best_three_splits(X, y, df['sample_name'].values, test_size=0.2, n_splits=100, random_state=42)

    ############################
    n= 1
    train_idx,test_idx = best_splits[n]
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    df_train, df_test = df3.loc[train_idx], df3.loc[test_idx]
    cells_train, cells_test = cells_np[train_idx], cells_np[test_idx]
    nucs_train , nucs_test = nucs_np[train_idx], nucs_np[test_idx]
    return   feature_names, X_train, X_test,y_train, y_test , df_train, df_test ,  cells_train, cells_test,  nucs_train , nucs_test