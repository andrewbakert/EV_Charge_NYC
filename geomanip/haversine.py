
import geopandas as gpd
import numpy as np
import pandas as pd

def haversine_1_way(coord1, coord2, units='metric', shapely=True):
    """
    Uses the Haversine equation to find the distance between a column of Shapely Point
    objects and a point in longitude and latitude coordinates
    
    Parameters
    ----------
    coord1 : GeoPandas GeoSeries
        column of Shapely Point objects
    coord2 : tuple or list
        coordinates for distance calculation, in format (longitude, latitude)
    units : str
        units for distance calculated, either metric or imperial
        metric units are meters, and imperial units are feet
    shapely: bool
        flag to determine if input column is of Shapely objects
    
    Returns
    -------
    Numpy array of distances from the GeoSeries to the coordinate
    """
    
    # Copy coordinate column to prevent alteration
    coord_col = coord1.copy()
    
    # Check if shapely flag is True. If so, convert to array of coords 
    # tranformed to latlong. If not, convert from list to array
    if shapely:
        coord_col = np.array(list(map(lambda x: x.coords.xy, 
                        coord_col.to_crs('EPSG:4326'))))
    else:
        coord_col = np.apply_along_axis(lambda x: np.array(list(x)), 0,  
                            np.array(coord_col))
        
    # Choose longitude and latitude arrays from columns
    lon1 = coord_col[:, 0].astype(float)
    lat1 = coord_col[:, 1].astype(float)
    
    # Expand list or tuple of coordinates
    lon2, lat2 = coord2
    
    # Create Haversine equation for future calculation, and set 
    # radius of Earth in meters
    hav = lambda theta: (1 - np.cos(theta)) / 2
    R = 6.3781e6
    
    # Check if units are imperial. If so, convert to feet.
    if units != 'metric':
        R *= 3.28084
        
    # Convert coordinates to radians
    lat1 *= (2 * np.pi / 360)
    lon1 *= (2 * np.pi / 360)
    lat2 *= (2 * np.pi / 360)
    lon2 *= (2 * np.pi / 360)   
    
    # Calculate distance
    d = 2 * R * np.arcsin(np.sqrt(hav(lat2 - lat1) +  
                np.cos(lat1) * np.cos(lat2) * hav(lon2 - lon1)))
    return d

def haversine_2_way(df1, df2, geo_col1='geometry', geo_col2='geometry', 
                    units='metric', dist_colname='distance', 
                    geo_name1='geometry1', geo_name2='geometry2',
                    shapely1=True, shapely2=True):
    """
    Calculates distance between Point objects between two dataframes.
    Combines dataframes along with distance.
    
    Parameters
    ----------
    df1 : GeoPandas GeoDataFrame
        first dataframe for comparison
    df2 : GeoPandas GeoDataFrame
        first dataframe for comparison
    geo_col1 : str
        name of geographic column or columns for first dataframe; 
        if two columns, follow format (longitude, latitude)
        (default is 'geometry')
    geo_col2 : str
        name of geographic column or columns for second dataframe; 
        if two columns, follow format (longitude, latitude)
        (default is 'geometry')
    units : str
        units of calculated distance, either metric or imperial
        (default is 'metric')
    dist_colname : str
        name for create column with distances
        (default is 'distance')   
    geo_name1 : str
        name for renaming of geometry column from first dataframe.
        Used to distingush from second geometry column when dataframes
        are combined.
        (default is 'geometry1')
    geo_name2 : str
        serves similar function to geo_name1, except destinguishes 
        second geometry column.
        (deafault is 'geometry2')
    shapely1 : bool
        flag for whether first column(s) is in Shapely format
        (default is True)
    shapely2 : bool
        flag for whether second column(s) is in Shapely format
        (default is True)
        
    Returns 
    -------
    Combined numpy array with elements and list of columns
    """
    
    # Copy first dataframe to prevent assignment
    df = df1.copy()
    
    # Check if there are two geo columns for df1.
    # If so, convert to one column and set index to last
    if len(geo_col1) == 2:
        df['latlong'] = df.apply(lambda x: 
                                 (x[geo_col1[0]], x[geo_col1[1]]),
                                axis=1)
        shapely1 = False
        geo_idx1 = -1
    else:
        # If only one column, find geo column idx
        geo_idx1 = np.argwhere(df1.columns == geo_col1)[0][0]
   
    # Same process for second df.
    if len(geo_col2) == 2:
        coord_col2 = df2.apply(lambda x: 
                                 (x[geo_col2[0]], x[geo_col2[1]]),
                                axis=1)
        shapely2 = False
    else:
        coord_col2 = df2[geo_col2].copy()
        
    # Check if geo column in first dataframe is in Shapely format.
    # If so, convert to latlong coords array.
    if shapely1:
        coord_col1 = np.array(list(map(lambda x: 
            (x.coords.xy[0][0], x.coords.xy[1][0]), 
            gpd.GeoSeries(df.iloc[:, geo_idx1]).to_crs('EPSG:4326'))))
    else:
        # If not, convert to numpy array
        coord_col1 = np.apply_along_axis(lambda x: np.array(list(x)), 0,  
                np.array(df.iloc[:, geo_idx1]))
        
    # Perform same analysis for second dataframe
    if shapely2:
        coord_col2 = np.array(list(map(lambda x: 
            (x.coords.xy[0][0], x.coords.xy[1][0]),
            gpd.GeoSeries(coord_col2).to_crs('EPSG:4326'))))
    else:
        coord_col2 = np.apply_along_axis(lambda x: np.array(list(x)), 0,  
                        np.array(coord_col2))
        
    # Repeat first and second coord to match up for distances. 
    # First coord is repeated from format [1, 2] to [1, 1, 2, 2]
    # Second coord is repeated from format [1, 2] to [1, 2, 1, 2]
    coord1 = np.repeat(coord_col1, coord_col2.shape[0], 0)
    coord2 = np.tile(coord_col2, (df.shape[0], 1))
    
    # Perform same repition for arrays
    arr1 = np.repeat(df.to_numpy(), coord_col2.shape[0], axis=0)
    arr2 = np.tile(df2.to_numpy(), (df.shape[0], 1))
    
    # Isolate lat and long columns and convert to float.
    lon1 = coord1[:, 0].astype(float)
    lat1 = coord1[:, 1].astype(float)
    lon2 = coord2[:, 0].astype(float)
    lat2 = coord2[:, 1].astype(float)
    
    # Create haversine function for further calcs.
    hav = lambda theta: (1 - np.cos(theta)) / 2
    
    # Radius of earth in meters. If not metric, convert to feet.
    R = 6.3781e6
    if units != 'metric':
        R *= 3.28084
        
    # Convert coords to radians.
    lat1 *= (2 * np.pi / 360)
    lon1 *= (2 * np.pi / 360)
    lat2 *= (2 * np.pi / 360)
    lon2 *= (2 * np.pi / 360)   
    
    # Calculate distance
    d = 2 * R * np.arcsin(np.sqrt(hav(lat2 - lat1) +  
                np.cos(lat1) * np.cos(lat2) * hav(lon2 - lon1)))
    
    # Concatenate arrays, then columns. Return both.
    arr1 = np.concatenate((arr1, d.reshape(-1, 1), arr2), axis=1)
    df1_copy = df1.rename({geo_col1: geo_name1}, axis=1)
    df2_copy = df2.rename({geo_col2: geo_name2}, axis=1)
    columns = df1_copy.columns.tolist() + [dist_colname] + df2_copy.columns.tolist()
    return pd.DataFrame(arr1, columns=columns)