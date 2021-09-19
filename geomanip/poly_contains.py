import pandas as pd
import numpy as np
import geopandas as gpd

def combine_geom_dfs(df1, df2, geo_col1='geometry', geo_col2='geometry', 
                     geo_name1='geometry1', geo_name2='geometry2', crs='EPSG:4326'):
    """
    Matches up Shapely Polygon column with Shapely Point column to find overlap.
    
    Parameters
    ----------
    df1 : GeoPandas GeoDataFrame
        first dataframe containing Polygon objects. The Polygon column must 
        be in Shapely Polygon format.
    df2 : GeoPandas GeoDataFrame
        secong dataframe containing Point objects. The Point column must 
        be in Shapely Point format.
    geo_col1 : str
        name of Polygon geometry column in first dataframe
        (default is 'geometry')
    geo_col1 : str
        name of Point geometry column in second dataframe
        (default is 'geometry')
    geo_name1 : str
        name for renaming of geometry column from first dataframe.
        Used to distingush from second geometry column when dataframes
        are combined.
        (default is 'geometry1')
    geo_name2 : str
        serves similar function to geo_name1, except destinguishes 
        second geometry column.
        (deafault is 'geometry2')
    crs : str
        CRS used for each object
        (default is 'EPSG:4326')
    
    Returns
    -------
    Combined Numpy array with matching dataframes based on which points are contained by the polygons.
    """
    # Copy dataframes to prevent alteration.
    df1_copy = df1.copy()
    df2_copy = df2.copy()

    # Convert both dataframes to desired crs.
    df1_copy = df1_copy.explode()
    df1_copy = df1_copy.to_crs(crs)
    df2_copy = df2_copy.to_crs(crs)

    # Find idx of specified geo columns. Used to subset.
    geo_col1_idx = np.where(df1_copy.columns == geo_col1)[0][0]
    geo_col2_idx = np.where(df2_copy.columns == geo_col2)[0][0]

    # Convert dataframes to arrays.
    arr1 = np.array(df1_copy)
    arr2 = np.array(df2_copy)
    
    # Repeat arrays to match up. First array is repeated with format from [1, 2] to [1, 1, 2, 2].
    # Second array formatted from [1, 2] to [1, 2, 1, 2].
    arr1_rep = np.repeat(arr1, arr2.shape[0], axis=0)
    arr2_rep = np.tile(arr2, (arr1.shape[0], 1))
    
    # Vectorize Shapely contains method and apply to geometry columns of arrays.
    contains_vec = np.vectorize(lambda x, y: x.contains(y))
    contains_arr = contains_vec(arr1_rep[:, geo_col1_idx], arr2_rep[:, geo_col2_idx])
    
    # Filter both arrays by the contains array boolean column.
    arr1_comb = arr1_rep[contains_arr]
    arr2_comb = arr2_rep[contains_arr]
    
    # Concatenate arrays and columns. Return both.
    arr_comb = np.concatenate([arr1_comb, arr2_comb], axis=1)
    df1_copy.rename({geo_col1: geo_name1}, axis=1, inplace=True)
    df2_copy.rename({geo_col2: geo_name2}, axis=1, inplace=True)
    columns = df1_copy.columns.tolist() + df2_copy.columns.tolist()
    return pd.DataFrame(arr_comb, columns=columns)
