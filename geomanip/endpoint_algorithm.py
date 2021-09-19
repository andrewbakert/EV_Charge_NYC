import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import MultiLineString, Polygon, Point, LineString
import os

class GeneratePoints:
    """
    A class used to generate projected routes travelled based on dataframe of 
    Shapely MultiLineString objects and traffic counts by station.
    
    Attributes
    ----------
    station_df : GeoDataFrame
        A Geopandas dataframe containing traffic data formatted with
        direction column, station label column, traffic total column,
        and column of MultiLineString geometries
    direction_col : str
        Column in dataframe with federal directions of routed. Must be integers 
        in format:
        1 - North, 3 - East, 5 - South, 7 - West
    station_col : str
        Column in dataframe containing ids of stations for short counts
    traffic_col : str
        Column in dataframe containing traffic counts
    geo_col : str
        Column in dataframe containing Shapely MultiLineString geometries
    straight_mult : float
        Multiplier applied to traffic flowing in same direction as origin
    traffic_exp : float
        Exponent used to adjust traffic counts and favor directions with more 
        traffic  
    
    Methods
    -------
    segments(curve)
        Helper method for prep_df method used to extract LineStrings objects from
        MutliLineString.
    create_circle(point, radius=100, num_sides=100)
        Helper method used to create a circle of a given radius around a point.
    find_poly_contains(line1, line2, radius=100, num_sides=100)
        Finds whether end of line intersects with polygon formed around beginning
        of another line. Used to find next segments.
    set_direction(line, dir_num)
        Used the direction of a line to either maintain or reverse its coordinates. 
        Needed to distinguish short counts of differing directions at the same
        station.
    prep_df()
        Used to prepare dataframe for analysis of path predictions
    distance_line_point(coords1, point, diff=1e-6)
        Finds distance between line segment and given point
    find_next_step(station_df, seed)
        Used seed station and direction to find next step of possible routes
    build_pred_steps(seed, min_perc=0.01)
        Loops through find_next_step using given seed until route ends with 
        percent of traffic carried less than min_perc
    generate_traffic_pred(seed, min_perc=0.01)
        Creates dataframe from results for visualization and manipulation
    """
    
    
    
    def __init__(self, station_df, radius=100, 
                 direction_col='FederalDirection', station_col='RCSTA', 
                 traffic_col='AvgWeekdayInterval_9', geo_col='geometry_y',
                 straight_mult=1.5, traffic_exp=1.5):
        self.station_df = station_df
        self.radius = radius
        self.direction_col = direction_col
        self.station_col = station_col
        self.traffic_col = traffic_col
        self.geo_col = geo_col
        self.straight_mult = straight_mult
        self.traffic_exp = traffic_exp
        
    def segments(self, curve):
        """
        Seperates MultiLineString into seperate LineString objects containing segments, 
        then combines back to MultiLineString
        
        Parameters
        ----------
        curve : MultiLineString object
            Line to be broken into segments
            
        Returns
        -------
        MultiLineString object containing segments
        """
        curve = [y for x in curve for y in x.coords]
        return MultiLineString(list(map(LineString, zip(curve[:-1], curve[1:]))))   
        
    def create_circle(self, point, radius=100, num_sides=100):
        """
        Creates circle of given radius surrounding point
        
        Parameters
        ----------
        point : tuple
            Point around which to build circle
        radius : float
            Radius of circle built
            Default is 100
        num_sides : int
            Number of sides in Polygon object. Used to estimate circular shape
            Default is 100
        
        Returns
        -------
        Shapely Polygon object with approximation of circle
        """
        
        # Unpack tuple of coordinates
        x, y = point
        
        # Create coordinates of circle approximation using trigonometric
        # properties
        circle = [(radius * np.cos(angle) + x, radius * np.sin(angle) + y) 
                  for angle in np.linspace(0, 2*np.pi, num_sides)]
        return Polygon(circle)

    def find_poly_contains(self, line1, line2, radius=100, num_sides=100):
        """
        Finds if a circle created around the endpoint of one line intersects with
        another line
        
        Parameters
        ----------
        line1 : Shapely MultiLineString object
            First line for comparison. Circle is drawn around last point of this line
        line2 : Shapely MultiLineString object
            Second line for comparison. Used to determine if first line intersects
        radius : float
            Radius drawn around last coords of first line
            Default is 100
        num_sides : int
            Number of sides used to approximate circle
            
        Returns
        -------
        Boolean value of whether circle drawn around final coordinate of first line
        intersects with second line.
        """
        coords = [x.coords.xy for x in line1]
        point = (coords[-1][0][-1], coords[-1][1][-1])
        return self.create_circle(point, radius, num_sides).intersects(line2)

    def set_direction(self, line, dir_num):
        """
        Adjusts direction of a MultiLineString object to match given direction code.
        
        Parameters
        ----------
        line : Shapely MultiLineString object
            Line that is either kept or reversed in direction
        dir_num : int
            Direction of traffic at segment. Can be any of following:
            1 - North, 3 - East, 5 - South, 7 - West
        
        Returns
        -------
        Shapely MultiLineString object, either reversed in direction or not.   
        """
        
        # Extract coords from line, seperate into x and y
        coords = [x.coords.xy for x in line]
        x_coords = [x[0] for x in coords]
        y_coords = [x[1] for x in coords]
        
        # Initialize boolean variable of whether to switch direction
        switch = False
        
        # If the direction is north, check that the first y coordinate is greater than the last.
        # If so, set switch variable to true.
        if dir_num == 1:
            if y_coords[0][0] > y_coords[-1][-1]:
                switch = True
                
        # If the direction is east, check that the first x coordinate is greater than the last.
        # If so, set switch variable to true.
        elif dir_num == 3:
            if x_coords[0][0] > x_coords[-1][-1]:
                switch = True
                
        # If the direction is south, check that the first y coordinate is less than the last.
        # If so, set switch variable to true.
        elif dir_num == 5:
            if y_coords[0][0] < y_coords[-1][-1]:
                switch = True
                
        # If the direction is west, check that the first x coordinate is less than the last.
        # If so, set switch variable to true.
        else:
            if x_coords[0][0] < x_coords[-1][-1]:
                switch = True
                
        # If the line needs to be switched, switch cords and recreate into MultiLineString object.
        if switch == True:
            x_coords = [x[::-1] for x in x_coords][::-1]
            y_coords = [y[::-1] for y in y_coords][::-1]
            line = MultiLineString([list(zip(x_coords[i], y_coords[i])) for i in range(len(y_coords))])
        return line
    
    def prep_df(self):
        """
        Prepare traffic dataframe for analysis of path prediction
        
        Returns
        -------
        Prepared dataframe
        """
        
        # Group by station column and direction collumn, then sum traffic and keep geometry column.
        traffic_part = (self.station_df
            .groupby([self.station_col, self.direction_col])[[self.traffic_col, self.geo_col]]
            .agg({self.traffic_col: 'sum', self.geo_col: 'first'})
            .reset_index()
            )
        
        # Adjust column names in dataframe and set direction using set_direction method
        traffic_part.columns = ['station', 'direction', 'traffic', 'geometry']
        traffic_part['geometry'] = traffic_part.apply(lambda x: self.set_direction(x['geometry'], x['direction']), axis=1)
        
        # Reform into GeoDataFrame
        traffic_part = gpd.GeoDataFrame(traffic_part, geometry='geometry')
        
        # Break geometry into segments and set this as primary geometry.
        traffic_part['geometry_line_seg'] = traffic_part['geometry'].map(lambda x: self.segments(x))
        traffic_part.rename({'geometry': 'geom_line', 'geometry_line_seg': 'geometry'}, axis=1, inplace=True)
        traffic_part = gpd.GeoDataFrame(traffic_part, geometry='geometry')
        
        # Separate first line geometry in order to compare to previous segment. Explode segment geometries.
        traffic_part['first_line_geom'] = traffic_part['geometry'].map(lambda x: x[0])
        traffic_part = traffic_part.explode().reset_index().drop(['level_0', 'level_1'], axis=1)
        
        # Extract line segment coordinates. Used to determine distance from previous line endpoint to line segment.
        self.points_geom = np.stack(traffic_part['geometry'].map(lambda x: x.coords.xy))
        self.traffic_part = traffic_part
        return traffic_part
    
    def distance_line_point(self, coords1, point, diff=1e-6):
        """
        Determine the distance from a line segment to a point.
        Function is vectorized for improved performance.
        
        Parameters
        ----------
        coords1 : nparray
            Set of coordinates for line segments
        point : tuple
            Point of x and y coordinates for comparison
        diff : float
            Value added to denominators to prevent division by zero
            
        Returns
        -------
        Array of distances from line segments to point
        """
        
        # Extract coordinates of points on segments from coordinates arrray.
        x1 = coords1[:, 0, 0]
        y1 = coords1[:, 1, 0]
        x2 = coords1[:, 0, 1]
        y2 = coords1[:, 1, 1]
        
        # Unpack tuple to x and y coords.
        x3, y3 = point
        
        # Calculate slope. Used to predict x coordinate of point along line where 
        # point's y coordinate lies.
        m = (y2 - y1 + diff) / (x2 - x1 + diff)
        
        # Calculate contants in equation of a line, i.e. ax + by + c = 0
        a = -m
        b = 1
        c = m * x1 - y1
        
        # Determine x coordinate along line where y coordinate equal to that of point.
        x = (y3 - y1) / m + x1
        
        # Compare x coordinate along a line to determine if point is out of range of lines perpendicular to segment.
        # If greater or less, calculate Euclidian distance at either end. If not, use equation of a line perpendicular to 
        # a point.
        d = np.where(np.greater(x, np.max([x1, x2], axis=0)) | np.less(x, np.min([x1, x2], axis=0)),
                     np.where((np.less(x, np.min([x1, x2], axis=0)) & (x1 == np.min([x1, x2], axis=0)))|
                              (np.greater(x, np.max([x1, x2], axis=0)) & (x1 == np.max([x1, x2], axis=0))), 
                             ((x3 - x1) ** 2 + (y3 - y1) ** 2) ** 0.5,
                             ((x3 - x2) ** 2 + (y3 - y2) ** 2) ** 0.5
                             ), np.abs(a * x3 + b * y3 + c) / ((a**2 + b**2)**0.5 + diff))
        return d

    def find_next_step(self, station_df, seed):
        """
        Find next options for theoretical driver given current station and direction
        
        Parameters
        ----------
        station_df : GeoDataFrame
            Dataframe containing station numbers and directions as well as geometries of points.
        seed : list
            List of dictionaries containing prior geometry, traffic percentage, and station labels
            
        Returns
        -------
        List of dictionaries for next available steps in simulated trip  
        """
        # Choose last station in seed route.
        part_seed = seed[-1]
        
        # Isolate points at end of seed route.
        point = (part_seed['geometry'][-1].coords.xy[0][-1], part_seed['geometry'][-1].coords.xy[1][-1])
        
        # Calculate distance from point at end of route to all line segements in the dataframe.
        # Used to select only 50 closest points to reduce runtime.
        station_df['dist'] = self.distance_line_point(self.points_geom, point)
        traffic_closest = station_df.sort_values('dist').iloc[:50]
        
        # Check if circle of given radius intersects with line segment. 
        traffic_closest['contains_point'] = traffic_closest['geometry'].intersects(
            self.create_circle(point, radius=self.radius))
        
        # Check if circle of given radius intersects with first segment in station line.
        # Differentiated because some segments intersect at points in the middle of line,
        # while others intersect at beginning of line, which indicates beginning of line directed
        # away from end of previous line.
        traffic_closest['contains_first'] = traffic_closest['first_line_geom'].intersects(
            self.create_circle(point, radius=self.radius))
        
        # Choose only stations and directions that have segments bordering point.
        next_step = traffic_closest.loc[traffic_closest['contains_point']]
        next_step_first = traffic_closest.loc[traffic_closest['contains_first']]
        
        # Filter out all routes that circle back to original station.
        next_step = next_step.loc[~next_step['station'].isin([x['station'] for x in seed])]
        next_step_first = next_step_first.loc[~next_step_first['station'].isin([x['station'] for x in seed])]
        
        # Only keep points with unique station and direction.
        next_step_first = next_step_first.drop_duplicates(['station', 'direction'])
        next_step = next_step.drop_duplicates(['station', 'direction'])
        
        # Find stations that do not intersect with the first segment, but do intersect with the line.
        next_step = next_step.loc[~next_step['station'].isin(next_step_first['station'])]
        
        # Concatenate to form all posibilities.
        full_next_step = pd.concat([next_step_first, next_step], axis=0)
        
        # Adjust traffic, by favoring routes in same direction and/or by favoring more trafficked directions
        full_next_step['traffic'] = full_next_step['traffic'] ** self.traffic_exp
        full_next_step['traffic'] = full_next_step.apply(lambda x: 
                                        x['traffic'] * self.straight_mult if x['direction'] == part_seed['direction']
                                                    else x['traffic'], axis=1)
        
        # Calculate percentage that travel along each route based on traffic.
        full_next_step['traffic_perc'] = full_next_step['traffic'].div(full_next_step['traffic'].sum())
        
        # Concatenate to form additional steps.
        steps = [[part_seed, {'station': x[1]['station'], 'direction': x[1]['direction'], 
            'traffic_perc': x[1]['traffic_perc'] * part_seed['traffic_perc'], 'geometry': x[1]['geom_line']}] 
            for x in full_next_step.iterrows()]
        if not steps:
            steps = ([[part_seed.copy()]], True)
        return steps

    def build_pred_steps(self, seed, min_perc=0.01):
        """
        Build full list of predicted steps given seed.
        
        Parameters
        ----------
        seed : dict
            Dictionary containing traffic, station number, station direction, and geometry of initial route.
        min_perc : float
            Minimum percentage of total traffic flow after which route is terminated
            
        Returns
        -------
        List of dictionaries containing possible routes
        """
        
        seed['geometry'] = self.set_direction(seed['geometry'], seed['direction'])
        # Check if formatted traffic dictionary is already created. If not, create.
        if not hasattr(self,'traffic_part'):
            self.traffic_part = self.prep_df()
            
        # Copy seed, add initial traffic percentage as 1, and add as list of dictionary,
        # As required for find_next_step.
        new_seed = seed.copy()
        new_seed.pop('traffic', None)
        new_seed['traffic_perc'] = 1
        steps = [[new_seed]]
        
        # Initialize list with all final steps and create a while loop that will run until all routes
        # are below the minimum percentage in traffic.
        final_steps = []
        while True:
            new_steps = []
            final_step = []
            
            # Loop through previous steps and find next steps.
            for step in steps:
                next_step = self.find_next_step(self.traffic_part, step)
                
                # If there is no next step because there is no route option, add route to final steps.
                if type(next_step) == tuple:
                    final_step.extend([step[:-1] + x for x in next_step[0]])
                    continue
                new_step = []
                final_step = []
                
                # Loop through next step found. Check if the traffic percentage is below minimum.
                # If so, add to final steps. If not, add to new steps to continue iteration.
                for part_step in next_step:
                    if part_step[-1]['traffic_perc'] < min_perc:
                        final_step.append(step[:-1] + part_step)
                    else:
                        new_step.append(step[:-1] + part_step)
                new_steps.extend(new_step)
                final_steps.extend(final_step)
            
            # If all routes have been added to final steps and new steps is empty, break loop.
            if not new_steps:
                break
            # Copy new steps for next iteration.
            steps = new_steps.copy()
        
        # Loop through all steps to find traffic based on initial seed traffic and traffic percentage.
        traffic_steps = []
        for step in final_steps:
            part_steps = []
            for part in step:
                part['traffic'] = part['traffic_perc'] * seed['traffic']
                part_steps.append(part)
            traffic_steps.append(part_steps) 
            
        # Save steps as attribute for following function.
        self.traffic_steps = traffic_steps
        return traffic_steps
    
    def generate_traffic_pred(self, seed, min_perc=0.01):
        """
        Build GeoDataFrame based on found steps.
        
        Parameters
        ----------
        seed : dict
            Dictionary containing traffic, station number, station direction, and geometry of initial route.
        min_perc : float
            Minimum percentage of total traffic flow after which route is terminated
            
        Returns
        -------
        GeoDataFrame of route prediction results
        """
        
        # Check if traffic steps have been created. If not, create them.
        if not hasattr(self,'traffic_steps'):
            self.traffic_steps = self.build_pred_steps(seed, min_perc=min_perc)
            
        # Initialize GeoDataFrame, then concatenate steps.
        full_df = gpd.GeoDataFrame()
        for step in self.traffic_steps:
            df = gpd.GeoDataFrame(step, geometry='geometry')
            full_df = pd.concat([full_df, df], axis=0)
        
        # Group by station and direction to find total traffic percentage and traffic.
        full_df = (full_df
             .groupby(['station', 'direction'])[['traffic_perc', 'traffic', 'geometry']]
             .agg({'traffic_perc': 'sum', 'traffic': 'sum', 'geometry': 'first'})
             .reset_index()
             .sort_values('traffic_perc', ascending=False))
    
        # Divide traffic by maximum traffic percentage, which is the number of routes.
        full_df['traffic'] = full_df['traffic'].map(lambda x: x / full_df['traffic_perc'].max())
        
        # Calculate adjusted traffic percentage.
        full_df['traffic_perc'] = full_df['traffic_perc'].div(full_df['traffic_perc'].max())
        full_df = gpd.GeoDataFrame(full_df, geometry='geometry')
        return full_df
    
    def build_full_df(self, full_seed_df, min_perc=0.01, fp='', crs='EPSG:32618'):
        """
        Generates GeoDataFrame with results from multiple seeds in form of dataframe.
        
        Parameters
        ----------
        full_seed_df : GeoDataFrame
            Dataframe with information on input seeds. Must contain four columns:
            station, direction, geometry, and traffic.
            Station contains station id numbers.
            Direction contains federal direction.
            Geometry contains LineString geometry of segment.
            Traffic contains traffic along segment.
        min_perc : float
            Minimum percentage in decimal form of total traffic at which algorithm completes.
        fp : string
            Filepath to which dataframe will be saved.
            Default is empty string.
        crs : string
            The CRS used for the GeoDataFrame.
            Default is 'EPSG:32618', the crs for New York City
            
        Returns
        -------
        GeoDataFrame with results         
        """
        if os.path.exists(fp):
            seed_df = gpd.read_file(fp)
            seed_df.crs = crs

        # Otherwise, create geojson file and dataframe.
        else:
            # Initialize class instance as None.
            results = []
            
            # Loop through seeds to evaluate different inputs.
            for seed in full_seed_df.iterrows():
                seed = seed[1].to_dict()

                # Build list of dictionaries for steps.
                result = self.build_pred_steps(seed, min_perc=0.005)
                seed_result = {'result': result}

                print('Station: {}'.format(seed['station']))
                print('\n')

                # Append results to list of results.
                results.append(seed_result)

            # Create list of dictionaries to convert to geojson file.
            full_dcts = []
            for result in results:
                i = 1
                for seed in result['result']:
                    j = 1
                    for step in seed:
                        part_dct = {'route': i,
                                    'part': j,
                                    'seed_station': seed[0]['station'],
                                    'seed_direction': seed[0]['direction'],
                                    'seed_traffic': seed[0]['traffic'],
                                    'station': step['station'],
                                    'direction': step['direction'],
                                    'traffic': step['traffic'],
                                    'traffic_perc': step['traffic_perc'],
                                    'geometry': step['geometry']
                                   }
                        full_dcts.append(part_dct)
                        j += 1 
                    i += 1

            # Convert results to GeoDataFrame, then save as geojson.
            seed_df = gpd.GeoDataFrame(full_dcts, geometry='geometry', crs='EPSG:32618')
            seed_df.crs = crs
            if fp:
                with open(fp, 'w') as traffic:
                    traffic.write(seed_df.to_json())
        return seed_df

def find_dist_ratio(segs, traffic_exp=1):
    """
    Creates metric for evaluation of trip prediction model. 
    Used to find how straight routes are.
    
    Parameters
    ----------
    segs : list
        List of dictionaries of steps for routes
    traffic_exp : float
        Factor to give preference in weighting to heavily trafficked routes
        Default is 1
    
    Returns
    -------
    Float containg weighted average ratio between highway distance travelled
    and direct distance. 
    """
    
    # Initialize metric value
    full_metric = 0
    for seg in segs:
        
        # Find coordinates of beginning of route.
        seg1 = seg[0]['geometry'][0].coords.xy
        x1 = seg1[0][0]
        y1 = seg1[1][0]
        
        # Find coordinates of end of route.
        seg2 = seg[-1]['geometry'][-1].coords.xy
        x2 = seg2[0][-1]
        y2 = seg2[1][-1]
        
        # Find Euclidian distance from beginning to end.
        dist = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        
        # Find total distance travelled along route.
        route_dist = sum([x['geometry'].length for x in seg])
        
        # Find ratio of route to Euclidian distance weighted by traffic percentage.
        metric = (dist / route_dist) * (seg[-1]['traffic_perc'] ** traffic_exp)
        full_metric += metric
        
    # Return weighted values divided by total traffic percentage to get weighted average.
    return full_metric / sum(seg[-1]['traffic_perc'] ** traffic_exp for seg in segs)

def find_traffic_ratio(segs, traffic_df, traffic_exp=1):
    """
    Creates metric for evaluation of trip prediciton model.
    Used to determine how trafficked routes are that are produced by the model.
    
    Parameters
    ----------
    segs : list
        List of dictionaries of steps for routes
    traffic_df : GeoDataFrame
        Dataframe containing traffic for each segment
    traffic_exp : float
        Factor to give preference in weighting to heavily trafficked routes
    
    Returns
    -------
    Float containing weighted average traffic.
    """
    
    # Initialize traffic distance, traffic percentage, and total traffic.
    traffic_perc_sum = 0
    traffic_dist_sum = 0
    traffic_sum = 0
    
    # Loop through all parts of the routes to find traffic.
    for seg in segs:
        for part in seg:
            
            # Find traffic for chosen section alone.
            traffic = traffic_df.loc[(traffic_df['station'] == part['station']) & 
                                     (traffic_df['direction'] == part['direction'])].iloc[0]['traffic']
            
            # Find coords of beginning and end of traffic segment to calculate distance.
            seg1 = part['geometry'][0].coords.xy
            x1 = seg1[0][0]
            y1 = seg1[1][0]
            seg2 = part['geometry'][-1].coords.xy
            x2 = seg2[0][-1]
            y2 = seg2[1][-1]
            dist = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
            
            # Multiple traffic by distance covered and traffic percentage.
            traffic_sum += traffic * dist * part['traffic_perc'] ** traffic_exp
            traffic_dist_sum += dist
            traffic_perc_sum += part['traffic_perc'] ** traffic_exp
            
    # Calculated the average traffic weighted by distance covered and proportion of total.
    traffic_avg = traffic_sum / traffic_dist_sum / traffic_perc_sum
    return traffic_avg

def find_avg_distance(segs, traffic_exp=1):
    """
    Creates metric for evaluation of trip prediciton model.
    Used to determine average distance from beginning coordinate for each segment travelled.
    
    Parameters
    ----------
    segs : list
        List of dictionaries of steps for routes
    traffic_exp : float
        Factor to give preference in weighting to heavily trafficked routes
        Default is 1
        
    Return
    ------
    Float containing weighted average distance.
    """
    
    # Initialize distance and traffic percentage.
    total_dist_traffic = 0
    total_traffic_perc = 0
    
    # Get coords of initial point.
    coords1 = segs[0][0]['geometry'][0].coords.xy
    x1 = coords1[0][0]
    y1 = coords1[1][0]
    
    # Loop through all parts of all routed to obtain distances
    for seg in segs:
        for part in seg:
            
            # Extract coords of end of each segment.
            coords2 = part['geometry'][-1].coords.xy
            x2 = coords2[0][-1]
            y2 = coords2[1][-1]
            
            # Calculate distance between end of segment and start of routes
            dist = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
            
            # Weight distance by traffic percentage
            dist_traffic = dist * part['traffic_perc'] ** traffic_exp
            total_traffic_perc += part['traffic_perc'] ** traffic_exp
            total_dist_traffic += dist_traffic
    
    # Calculated weighted average of travel distance.
    avg_distance = total_dist_traffic / total_traffic_perc
    return avg_distance