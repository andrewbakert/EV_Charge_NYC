import pandas as pd
import numpy as np

def inherently_superior(df):
    """
    Find rows in a dataframe with all values 'inherently superior',
    meaning that all values for certain metrics are as high or higher
    then for all other rows.
    
    Parameters
    ----------
    df : DataFrame
        Pandas dataframe containing the columns to be compared. The columns
        should be in a format in which higher values are superior.
        
    Returns
    -------
    DataFrame with index of best values and values compared.
    """
    
    # Copy dataframe to prevent altering the columns. 
    df_copy = df.copy()
    
    # Reset index to reference location of values. Also, convert to numpy.
    df_copy.reset_index(inplace=True)
    arr = df_copy.values
    
    # Repeat and tile the array for comparison. Given indices [1, 2], arr1 is
    # in format [1, 1, 2, 2], and arr2 is in format [1, 2, 1, 2].
    arr1 = np.repeat(arr, arr.shape[0], axis=0)
    arr2 = np.tile(arr, (arr.shape[0], 1))
    
    # Check if any values are greater than for other rows.
    any_arr = np.all(arr1[:, 1:] >= arr2[:, 1:], axis=1)   
    
    # Adjust array so that all points at which a row is being compared to itself 
    # are labeled as superior.
    same_idx = np.array(range(0, len(any_arr), arr.shape[0])) + np.array(range(arr.shape[0]))
    any_arr[same_idx] = 1
    
    # Concatenate arr1 and array with superior labels.
    arr1_any = np.concatenate([arr1, any_arr.reshape(-1, 1)], axis=1)
    
    # Split data at unique indices. Used to check if greater than all other rows.
    splits = np.array(np.split(arr1_any, np.unique(arr1[:, 0], return_index=True)[1][1:]))
    perc_sup = np.mean(splits[:, :, -1], axis=1)
    idx = np.all(splits[:, :, -1], axis=1)
    
    # Choose superior data idx and create dataframe.
    columns = df_copy.columns.tolist() + ['perc_sup', 'fully_sup']
    data = np.concatenate([arr, perc_sup.reshape(-1, 1), idx.reshape(-1, 1)], axis=1)
    arr_df = pd.DataFrame(data, columns=columns)
    arr_df.drop('index', axis=1, inplace=True)
    arr_df['fully_sup'] = arr_df['fully_sup'].astype(bool)
    return arr_df
