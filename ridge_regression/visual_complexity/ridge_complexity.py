import glob
import os
from tqdm import tqdm
import numpy as np
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import IncrementalPCA

from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, ShuffleSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from PIL import Image
from scipy.stats import entropy
import skimage
from skimage.color import rgb2gray

from scipy.stats import pearsonr, ttest_1samp, sem
import warnings
from sklearn.metrics import make_scorer,r2_score
from sklearn.datasets import make_regression


from scipy.stats import ttest_1samp


def aggregate_df_by_layer(df):
    """
    Aggregates a single DataFrame by layer, averaging R values and computing combined significance,
    ensuring scalar values for each column.
    """
    aggregated_data = []

    for layer, group in df.groupby('Layer'):
        mean_r = group['R'].mean()
        t_stat, significance = ttest_1samp(group['R'], 0)
        
        # Assuming all rows within a single DataFrame have the same ROI
        common_roi_name = find_common_roi_name(group['ROI'].tolist())

        layer_data = {
            'ROI': common_roi_name,
            'Layer': layer,
            'Model': group['Model'].iloc[0],
            'R': mean_r,  # Use scalar value
            '%R2': mean_r ** 2,  # Use scalar value for %R2, computed from mean_r
            'Significance': significance,  # Use scalar value
            'SEM': group['R'].sem(),  # Use scalar value for SEM, if needed
            'LNC': np.nan,  # Placeholder for LNC, adjust as needed
            'UNC': np.nan  # Placeholder for UNC, adjust as needed
        }

        aggregated_data.append(layer_data)

    return pd.DataFrame(aggregated_data)


def find_common_roi_name(names):
    """
    Identifies the common ROI name within a single DataFrame.
    """
    if len(names) == 1:
        return names[0]  # Directly return the name if there's only one

    split_names = [name.split('_') for name in names]
    common_parts = set(split_names[0]).intersection(*split_names[1:])
    common_roi_name = '_'.join(common_parts)
    return common_roi_name

def aggregate_layers(dataframes):
    """
    Processes each DataFrame independently to aggregate by layer, then combines the results.
    Each DataFrame represents its own ROI, maintaining a single aggregated value per layer.
    """
    # Ensure dataframes is a list
    if isinstance(dataframes, pd.DataFrame):
        dataframes = [dataframes]

    aggregated_dfs = []

    for df in dataframes:
        aggregated_df = aggregate_df_by_layer(df)
        aggregated_dfs.append(aggregated_df)

    # Combine aggregated results from all DataFrames
    final_df = pd.concat(aggregated_dfs, ignore_index=True)
    
    return final_df


def get_layers_ncondns(feat_path):
    """
    Extracts information about the number of layers, the list of layer names, and the number of conditions (images)
    from the npz files in the specified feature path.

    Parameters:
    - feat_path (str): Path to the directory containing npz files with model features.

    Returns:
    - num_layers (int): The number of layers found in the npz files.
    - layer_list (list of str): A list containing the names of the layers.
    - num_conds (int): The number of conditions (images) based on the number of npz files in the directory.
    """
    
    # Find all npz files in the specified directory
    activations = glob.glob(feat_path + "/*.npz")
    
    # Count the number of npz files as the number of conditions (images)
    num_condns = len(activations)
    
    # Load the first npz file to extract layer information
    feat = np.load(activations[0], allow_pickle=True)

    num_layers = 0
    layer_list = []

    # Iterate through the keys in the npz file, ignoring metadata keys
    for key in feat:
        if "__" in key:  # key: __header__, __version__, __globals__
            continue
        else:
            num_layers += 1
            layer_list.append(key)  # collect all layer names ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8']

    return num_layers, layer_list, num_condns

def encode_layer(layer_id, n_components, batch_size, trn_Idx, tst_Idx, feat_path):
    """
    Encodes the layer activations using IncrementalPCA, for both training and test sets.

    Parameters:
    - layer_id (str): The layer name whose activations are to be encoded.
    - n_components (int): Number of components for PCA.
    - batch_size (int): Batch size for IncrementalPCA.
    - trn_Idx (list of int): Indices of the training set files.
    - tst_Idx (list of int): Indices of the test set files.
    - feat_path (str): Path to the directory containing npz files with model features.

    Returns:
    - pca_trn (numpy.ndarray): PCA-encoded features of the training set.
    - pca_tst (numpy.ndarray): PCA-encoded features of the test set.
    """
    activations = []
    feat_files = glob.glob(feat_path + '/*.npz')
    feat_files.sort()  # Ensure consistent order

    # Load a sample feature to check its dimensions after processing
    sample_feat = np.load(feat_files[0], allow_pickle=True)[layer_id]
    processed_sample_feat = np.mean(sample_feat, axis=1).flatten()

    # Determine whether to use PCA based on the dimensionality of the processed features
    #use_pca = processed_sample_feat.ndim > 1 or (processed_sample_feat.ndim == 1 and processed_sample_feat.shape[0] > 1)
    use_pca = False

    if use_pca:

        print(use_pca)
        pca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
        for jj,ii in enumerate(trn_Idx):  # for each datafile for the current layer
            feat = np.load(feat_files[ii], allow_pickle=True)  # get activations of the current layer
            activations.append(np.mean(feat[layer_id], axis=1).flatten())
        
            # Partially fit the PCA model in batches
            if ((jj + 1) % batch_size) == 0 or (jj + 1) == len(trn_Idx):
                pca.partial_fit(np.stack(activations[-batch_size:],axis=0))
                
        # Transform the training set using the trained PCA model
        pca_trn = pca.transform(np.stack(activations,axis=0))
        
        # Repeat the process for the test set
        activations = []
        for ii in tst_Idx:  # for each datafile for the current layer
            feat = np.load(feat_files[ii], allow_pickle=True)  # get activations of the current layer
            activations.append(np.mean(feat[layer_id], axis=1).flatten())
        pca_tst = pca.transform(np.stack(activations,axis=0))

    else:
        print(use_pca)
        print("use no pca")
        # Directly use the activations without PCA transformation and ensure they are reshaped to 2D arrays
        
        #pca_trn = np.array([np.mean(np.load(feat_files[ii], allow_pickle=True)[layer_id], axis=1).flatten() for ii in trn_Idx]).reshape(-1, 1)
        #pca_tst = np.array([np.mean(np.load(feat_files[ii], allow_pickle=True)[layer_id], axis=1).flatten() for ii in tst_Idx]).reshape(-1, 1)

        trn = np.array([np.mean(np.load(feat_files[ii], allow_pickle=True)[layer_id], axis=1).flatten() for ii in trn_Idx])
        tst = np.array([np.mean(np.load(feat_files[ii], allow_pickle=True)[layer_id], axis=1).flatten() for ii in tst_Idx])
        
        print(trn.shape)
        print(tst.shape)
    return trn, tst


def pearson_corr(y_true, y_pred):
    # Compute Pearson correlation
    corr, _ = pearsonr(y_true, y_pred)
    return corr

def calculate_noise_ceiling(tst_y, n_splits=5, random_state=42):
    """
    Calculate the noise ceiling for the fMRI data.

    Args:
        tst_y (numpy.ndarray): fMRI test set data.
        n_splits (int): Number of splits for split-half reliability calculation.
        random_state (int): Seed for random operations to ensure reproducibility.

    Returns:
        noise_ceiling (float): Estimated noise ceiling.
    """
    np.random.seed(random_state)
    n_samples, n_voxels = tst_y.shape
    split_correlations = []

    for _ in range(n_splits):
        # Shuffle indices and split the data into two halves
        indices = np.random.permutation(n_samples)
        half_size = n_samples // 2
        split_1, split_2 = indices[:half_size], indices[half_size:half_size*2]

        # Compute responses for each half
        responses_1, responses_2 = tst_y[split_1], tst_y[split_2]

        # Calculate pairwise correlations for each voxel
        voxel_correlations = []
        for v in range(n_voxels):
            corr, _ = pearsonr(responses_1[:, v], responses_2[:, v])
            voxel_correlations.append(corr)
        
        # Store the mean correlation for this split
        split_correlations.append(np.mean(voxel_correlations))
    
    # Noise ceiling is the average of the split-half correlations
    noise_ceiling = np.mean(split_correlations)
    return noise_ceiling
 
# Function to calculate the entropy of an image
def calculate_entropy(image_path):
    # Open the image using Pillow
    img = skimage.io.imread(image_path)
    img = rgb2gray(img)
    entropy = skimage.measure.shannon_entropy(img)    
    return entropy  

def measure_complexity(subject):
    
    Entropy = []
    path = f"/scratch/nikiguo93/predify-n2b/NSD/Algonauts_NSD/subj0{subject}/training_split/training_images/"    
    image_files = glob.glob(path + '*.png')
    image_files.sort()
    for i in range(len(image_files)):
        entropy_value = calculate_entropy(image_files[i])
        Entropy.append(entropy_value)
    complexity_scores = np.asarray(Entropy)
    return complexity_scores
    
def get_testset_complexity(tst_Idx,tst_x,tst_y,complexity_scores):
    
    complexity_score = complexity_scores[tst_Idx]
    
    # Get the sorted indices
    sorted_indices = np.argsort(complexity_score)

    # Calculate the number of values for top 30% and lowest 30%
    n = len(complexity_score)
    top_30_percent_count = int(0.1 * n)

    # Get indices for the top 30% lowest values
    top_30_percent_lowest_indices = sorted_indices[:top_30_percent_count]

    # Get indices for the top 30% highest values
    top_30_percent_highest_indices = sorted_indices[-top_30_percent_count:]
    tst_x_high = tst_x[top_30_percent_highest_indices]
    tst_x_low = tst_x[top_30_percent_lowest_indices]
    tst_y_high = tst_y[top_30_percent_highest_indices]
    tst_y_low = tst_y[top_30_percent_lowest_indices]
    
    score_high = complexity_score[top_30_percent_highest_indices].mean()
    score_low = complexity_score[top_30_percent_lowest_indices].mean()
    return tst_x_high,tst_x_low,tst_y_high,tst_y_low,score_high,score_low   
    
    
def train_Ridgeregression_per_ROI(trn_x,tst_x_high,tst_x_low,trn_y,tst_y_high,tst_y_low):
    """
    Train a linear regression model for each ROI and compute correlation coefficients.

    Args:
        trn_x (numpy.ndarray): PCA-transformed training set activations.
        tst_x (numpy.ndarray): PCA-transformed test set activations.
        trn_y (numpy.ndarray): fMRI training set data.
        tst_y (numpy.ndarray): fMRI test set data.

    Returns:
        correlation_lst (numpy.ndarray): List of correlation coefficients for each ROI.
    """

    # Normalize features
    scaler = StandardScaler()
    trn_x = scaler.fit_transform(trn_x)
    tst_x_high = scaler.transform(tst_x_high)
    tst_x_low = scaler.transform(tst_x_low)
    
    #reg = LinearRegression().fit(trn_x, trn_y)
    reg = Ridge()
    param_grid = {'alpha': np.logspace(-2, 2, 5)}
    #sgd_reg = SGDRegressor(penalty='l2', alpha=alpha, max_iter=1000, tol=1e-3, random_state=42)
    #pipeline = make_pipeline(StandardScaler(), sgd_reg)

    # Define cross-validation strategies
    inner_cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=1)
    outer_cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=1)

    # Inner loop: hyperparameter tuning
    grid_search = GridSearchCV(estimator=reg, param_grid=param_grid, cv=inner_cv, n_jobs=-1)

    X_sample, _, y_sample, _ = train_test_split(trn_x, trn_y, test_size=0.5, random_state=1)

    # Outer loop: model evaluation
    nested_cv_scores = cross_val_score(grid_search, X=X_sample, y=y_sample, cv=outer_cv, n_jobs=-1)

    # Results
    print(f"Nested CV Mean R^2: {nested_cv_scores.mean():.3f} ± {nested_cv_scores.std():.3f}")

    
    print('for training:', trn_x.shape)
    print('for training:', trn_y.shape)
    
    #pipeline.fit(trn_x, trn_y)
    #reg.fit(trn_x, trn_y)

    # Train the best model on the full dataset
    grid_search.fit(X_sample, y_sample)
    best_params = grid_search.best_params_

    best_model = Ridge(**best_params)
    best_model.fit(trn_x, trn_y)

    # Evaluate the model
    #train_score = pipeline.score(trn_x, trn_y)
    #test_score = pipeline.score(tst_x, tst_y)

    #print(f"Training R^2 score: {train_score:.4f}")
    #print(f"Testing R^2 score: {test_score:.4f}")

    #y_prd = reg.predict(tst_x)
    y_prd_high = best_model.predict(tst_x_high)
    y_prd_low = best_model.predict(tst_x_low)


    correlation_lst_high = np.zeros(y_prd_high.shape[1])
    correlation_lst_low = np.zeros(y_prd_low.shape[1])
    
    for v in range(y_prd_high.shape[1]):
        correlation_lst_high[v] = pearsonr(y_prd_high[:,v], tst_y_high[:,v])[0]
        
    for v in range(y_prd_low.shape[1]):
        correlation_lst_low[v] = pearsonr(y_prd_low[:,v], tst_y_low[:,v])[0]
        
    return correlation_lst_high,correlation_lst_low


def Ridge_Encoding(feat_path, roi_path, model_name, subject,trn_tst_split=0.8, n_folds=3, n_components=100, batch_size=100, just_corr=True, return_correlations = False,random_state=14, save_path="Linear_Encoding_Results"):
    """
    Perform linear encoding analysis to relate model activations to fMRI data across multiple folds.

    Args:
        feat_path (str): Path to the directory containing model activation .npz files for multiple layers.
        roi_path (str or list): Path to the directory containing .npy fMRI data files for multiple ROIs.
        
            If we have a list of folders, each folders content will be summarized into one value. This is important if one folder contains data for the same ROI for different subjects
            
        model_name (str): Name of the model being analyzed (used for labeling in the output).
        trn_tst_split (float): Proportion of data to use for training (rest is used for testing).
        n_folds (int): Number of folds to split the data for cross-validation.
        n_components (int): Number of principal components to retain in PCA.
        batch_size (int): Batch size for Incremental PCA.
        just_corr (bool): If True, only correlation values are considered in analysis (currently not used in function body).
        return_correlations (bool): If True, return correlation values for each ROI and layer.
        random_state (int): Seed for random operations to ensure reproducibility.

    Returns:
        all_rois_df (pd.DataFrame): DataFrame summarizing the analysis results including correlations and statistical significance.
        corr_dict (dict): Dictionary containing correlation values for each layer and ROI (only if return_correlations is True).
    """
    
    # Check if its a list
    roi_paths = roi_path if isinstance(roi_path, list) else [roi_path]
    
    list_dataframes_high = []
    list_dataframes_low = []
    
    # Iterate through all folder paths
    for roi_path in tqdm(roi_paths):
        result_high, result_low = _ridge_encoding(feat_path, 
                                            roi_path, 
                                            model_name,
                                            subject,
                                            trn_tst_split=trn_tst_split, 
                                            n_folds=n_folds, 
                                            n_components=n_components, 
                                            batch_size=batch_size,
                                            just_corr=just_corr, 
                                            return_correlations=return_correlations,
                                            random_state=random_state)
        

        # If return_correlations is True, unpack additional returned values
        if return_correlations:
            result_dataframe_high, corr_dict_high = result_high
            result_dataframe_low, corr_dict_low = result_low
        else:
            result_dataframe_high = result_high
            result_dataframe_low = result_low
        
        print(result_dataframe_high)  
        print(result_dataframe_low)    

        # Collect dataframes in list
        list_dataframes_high.append(result_dataframe_high)
        list_dataframes_low.append(result_dataframe_low)
        
    # If just one dataframe, return it as it is, else aggregate the layers
    final_df_high = list_dataframes_high[0] if len(list_dataframes_high) == 1 else aggregate_layers(list_dataframes_high)
    final_df_low = list_dataframes_low[0] if len(list_dataframes_low) == 1 else aggregate_layers(list_dataframes_low)
        
    # Create the output folder if it doesn't exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    csv_file_path_high = f"{save_path}/{model_name}_high.csv"
    csv_file_path_low = f"{save_path}/{model_name}_low.csv"
    
    final_df_high.to_csv(csv_file_path_high, index=False)
    final_df_low.to_csv(csv_file_path_low, index=False)
    
    return final_df_high,final_df_low
        
        
        
        
def ridge_encoding(*args, **kwargs):
    warnings.warn(
        "The 'linear_encoding' function is deprecated and has been replaced by 'Linear_Encoding'. "
        "Please update your code to use the new function name, as this alias will be removed in a future version.",
        DeprecationWarning,
        stacklevel=2
    )
    return Ridge_Encoding(*args, **kwargs)   
        
    

def _ridge_encoding(feat_path, roi_path, model_name,subject, trn_tst_split=0.8, n_folds=3, n_components=100, batch_size=100, just_corr=True, return_correlations = False,random_state=14):
    """
    Perform linear encoding analysis to relate model activations to fMRI data across multiple folds.

    Args:
        feat_path (str): Path to the directory containing model activation .npz files for multiple layers.
        roi_path (str): Path to the directory containing .npy fMRI data files for multiple ROIs.
        model_name (str): Name of the model being analyzed (used for labeling in the output).
        trn_tst_split (float): Proportion of data to use for training (rest is used for testing).
        n_folds (int): Number of folds to split the data for cross-validation.
        n_components (int): Number of principal components to retain in PCA.
        batch_size (int): Batch size for Incremental PCA.
        just_corr (bool): If True, only correlation values are considered in analysis (currently not used in function body).
        return_correlations (bool): If True, return correlation values for each ROI and layer.
        random_state (int): Seed for random operations to ensure reproducibility.

    Returns:
        all_rois_df (pd.DataFrame): DataFrame summarizing the analysis results including correlations and statistical significance.
        corr_dict (dict): Dictionary containing correlation values for each layer and ROI (only if return_correlations is True).
    """
    
    # Initialize dictionaries to store results
    fold_dict_high = {}  # To store fold-wise results
    fold_dict_low = {} 
    corr_dict_high = {}  # To store correlations if requested
    corr_dict_low = {}
    complexity_scores_high = {}
    complexity_scores_low = {}
    noise_ceilings = {}
    
    # Check if roi_path is a list, if not, make it a list
    roi_paths = roi_path if isinstance(roi_path, list) else [roi_path]
    
    # Load feature files and get layer information
    feat_files = glob.glob(feat_path+'/*.npz')
    num_layers, layer_list, num_condns = get_layers_ncondns(feat_path)

    complexity_scores = measure_complexity(subject)
    
    # Create a tqdm object with an initial description
    pbar = tqdm(range(n_folds), desc="Initializing folds")
    
    # Loop over each fold for cross-validation
    for fold_ii in pbar:
        pbar.set_description(f"Processing fold {fold_ii + 1}/{n_folds}")
        
        # Set random seeds for reproducibility
        np.random.seed(fold_ii+random_state)
        random.seed(fold_ii+random_state)
        
        # Split the data indices into training and testing sets
        trn_Idx,tst_Idx = train_test_split(range(len(feat_files)),test_size=(1-trn_tst_split),train_size=trn_tst_split,random_state=fold_ii+random_state)
        
        # Process each layer of model activations
        for layer_id in layer_list:
            if layer_id not in fold_dict_high.keys():
                fold_dict_high[layer_id] = {}
                fold_dict_low[layer_id] = {}
                corr_dict_high[layer_id] = {}
                corr_dict_low[layer_id] = {}
                noise_ceilings[layer_id] = {}
                complexity_scores_high[layer_id] = {}
                complexity_scores_low[layer_id] = {}
            
            # Encode the current layer using PCA and split into training and testing sets
            pca_trn,pca_tst = encode_layer(layer_id, n_components, batch_size, trn_Idx, tst_Idx, feat_path)

            for roi_path in roi_paths:
                roi_files = []
            
                 # Check if the roi_path is a file or a directory
                if os.path.isfile(roi_path) and roi_path.endswith('.npy'):
                    # If it's a file, directly use it
                    roi_files.append(roi_path)
                elif os.path.isdir(roi_path):
                    # If it's a directory, list all .npy files within it
                    roi_files.extend(glob.glob(os.path.join(roi_path, '*.npy')))
                else:
                    print(f"Invalid ROI path: {roi_path}")
                    continue  # Skip this roi_path if it's neither a valid file nor a directory

                # Process each ROI's fMRI data
                if not roi_files:
                    print(f"No roi_files found in {roi_path}")
                    continue  # Skip to the next roi_path if no ROI files were found

                for roi_file in roi_files:
                    roi_name = os.path.basename(roi_file)[:-4]
                    if roi_name not in fold_dict_high[layer_id].keys():
                        fold_dict_high[layer_id][roi_name] = []
                        fold_dict_low[layer_id][roi_name] = []
                        corr_dict_high[layer_id][roi_name] = []
                        corr_dict_low[layer_id][roi_name] = []
                        noise_ceilings[layer_id][roi_name] = [] 
                        complexity_scores_high[layer_id][roi_name] = []
                        complexity_scores_low[layer_id][roi_name] = []
                        
                    # Load fMRI data for the current ROI and split into training and testing sets
                    fmri_data = np.load(os.path.join(roi_file))
                    fmri_trn,fmri_tst = fmri_data[trn_Idx],fmri_data[tst_Idx]
                    
                    # Train a linear regression model and compute correlations for the current ROI
                    pca_tst_high,pca_tst_low,fmri_tst_high,fmri_tst_low,score_high,score_low  = get_testset_complexity(tst_Idx,pca_tst,fmri_tst,complexity_scores)
                    
                    r_lst_high,r_lst_low = train_Ridgeregression_per_ROI(pca_trn,pca_tst_high,pca_tst_low,fmri_trn,fmri_tst_high,fmri_tst_low)
                    r_high = np.mean(r_lst_high) # Mean of all train test splits
                    r_low = np.mean(r_lst_low)
                
                    
                    # Store correlation results
                    if return_correlations:                   
                        corr_dict_high[layer_id][roi_name].append(r_lst_high)
                        corr_dict_low[layer_id][roi_name].append(r_lst_low)
                        if fold_ii == n_folds-1:
                            corr_dict_high[layer_id][roi_name] = np.mean(np.array(corr_dict_high[layer_id][roi_name], dtype=np.float16),axis=0)
                            corr_dict_low[layer_id][roi_name] = np.mean(np.array(corr_dict_low[layer_id][roi_name], dtype=np.float16),axis=0)
                    fold_dict_high[layer_id][roi_name].append(r_high)
                    fold_dict_low[layer_id][roi_name].append(r_low)

                    # Calculate noise ceiling for the current ROI on the test set
                    noise_ceiling = calculate_noise_ceiling(fmri_tst_high)
                    noise_ceilings[layer_id][roi_name].append(noise_ceiling)
                    
                    complexity_scores_high[layer_id][roi_name].append(score_high)
                    complexity_scores_low[layer_id][roi_name].append(score_low)
                
    # Compile all results into a DataFrame for easy analysis
    all_rois_df_high = pd.DataFrame(columns=['ROI', 'Layer', "Model", 'R', '%R2', 'Significance', 'SEM', 'LNC', 'UNC','Complexity_score'])
    for layer_id,layer_dict in fold_dict_high.items():
        for roi_name,r_lst in layer_dict.items():
            
            # Compute statistical significance of the correlations
            significance = ttest_1samp(r_lst, 0)[1]
            R = np.mean(r_lst)
            r_lst_array = np.array(r_lst)  # Convert the list to a NumPy array
            
            mean_noise_ceiling = np.mean(noise_ceilings[layer_id][roi_name])
            unc = mean_noise_ceiling - np.mean(r_lst)
            lnc = mean_noise_ceiling
            
            mean_score_high = np.mean(complexity_scores_high[layer_id][roi_name])
            
            output_dict = {"ROI":roi_name,
            "Layer": layer_id,
            "Model": model_name,
            "R": [R],
            "%R2": [R ** 2],
            "Significance": [significance],
            "SEM": [sem(r_lst_array)],
            "LNC": [lnc],
            "UNC": [unc],
            "Complexity_score": [mean_score_high]}
            layer_df = pd.DataFrame.from_dict(output_dict)
            all_rois_df_high = pd.concat([all_rois_df_high, layer_df], ignore_index=True)
            
    all_rois_df_low = pd.DataFrame(columns=['ROI', 'Layer', "Model", 'R', '%R2', 'Significance', 'SEM', 'LNC', 'UNC','Complexity_score'])
    for layer_id,layer_dict in fold_dict_low.items():
        for roi_name,r_lst in layer_dict.items():
            
            # Compute statistical significance of the correlations
            significance = ttest_1samp(r_lst, 0)[1]
            R = np.mean(r_lst)
            r_lst_array = np.array(r_lst)  # Convert the list to a NumPy array
            mean_noise_ceiling = np.mean(noise_ceilings[layer_id][roi_name])
            unc = mean_noise_ceiling - np.mean(r_lst)
            lnc = mean_noise_ceiling
            
            mean_score_low = np.mean(complexity_scores_low[layer_id][roi_name])
            
            output_dict = {"ROI":roi_name,
            "Layer": layer_id,
            "Model": model_name,
            "R": [R],
            "%R2": [R ** 2],
            "Significance": [significance],
            "SEM": [sem(r_lst_array)],
            "LNC": [lnc],
            "UNC": [unc],
            "Complexity_score": [mean_score_low]}
            layer_df = pd.DataFrame.from_dict(output_dict)
            all_rois_df_low = pd.concat([all_rois_df_low, layer_df], ignore_index=True)
            
    if return_correlations:
        return ([all_rois_df_high, corr_dict_high], [all_rois_df_low, corr_dict_low])
    
    return (all_rois_df_high, all_rois_df_low)
