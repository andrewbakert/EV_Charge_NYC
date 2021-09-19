
from shapely.geometry import Polygon, MultiPolygon
import numpy as np

def find_poly_coord(line_shape, dist, line_style='multi'):
    """
    Builds Shapely Polygon along line segments at a given distance 
    perpendicular to the segments. Must use .explode() on GeoPandas 
    multiline.
    
    Parmeters
    ---------
    line_shape : Shapely MultilineString
        multiline used as basis to build polygon
    dist : float
        distance perpendicular to multiline to draw polygon
        on each side
    line_style : str
        style of line, either 'multi' for MultilineString, 
        or 'single for LineString.
        
    Returns
    -------
    Shapely MultiPolygon shape following MultilineString shape
    at distance `dist`
    """
    if line_style == 'multi':
        # Generate xy coords from MultilineString object
        xy = [y for x in line_shape for y in list(zip(list(x.coords.xy[0]), list(x.coords.xy[1])))]
    else:
        # Generate xy coords from LineString object
        xy = list(zip(list(line_shape.coords.xy[0]), list(line_shape.coords.xy[1])))
    
    # Create empty list for Polygons, and loop through coords
    poly_coord = []
    for i in range(1, len(xy)):
        
        # Select x and y coords by line segment
        x1 = xy[i - 1][0]
        y1 = xy[i - 1][1]
        x2 = xy[i][0]
        y2 = xy[i][1]
        
        # If no change in x, only build polygon at distance from x
        if x2 == x1:
            poly = Polygon([(x1 + dist, y1), (x1 - dist, y1), 
                            (x2 - dist, y2), (x2 + dist, y2)])
        
        # If no change in y, only build polygon at distance from y
        elif y2 == y1:
            poly = Polygon([(x1, y1 + dist), (x1, y1 - dist), 
                            (x2, y2 - dist), (x2, y2 + dist)])
        else: 
            # Determine angle of segment from x-axis and get orthogonal
            # angle
            theta = np.arctan((y2 - y1) / (x2 - x1))
            phi = theta - (np.pi / 2)
            
            # Use trig properties to extract all coordinates of rectangle
            # along segment
            x1_new = np.cos(phi) * -dist + x1
            y1_new = np.sin(phi) * -dist + y1
            x2_new = np.cos(phi) * -dist + x2
            y2_new = np.sin(phi) * -dist + y2
            x3_new = np.cos(phi) * dist + x2
            y3_new = np.sin(phi) * dist + y2
            x4_new = np.cos(phi) * dist + x1
            y4_new = np.sin(phi) * dist + y1
            
            # Generate Shapely Polygon object from new coords
            poly = Polygon([(x1_new, y1_new), (x2_new, y2_new), 
                                    (x3_new, y3_new), (x4_new, y4_new)])
            
        # Append Polygon to list of all Polygons
        poly_coord.append(poly)
    
    # Return at MultiPolygon object
    return MultiPolygon(poly_coord)       