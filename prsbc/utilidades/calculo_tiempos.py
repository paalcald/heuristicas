import pandas as pd
import numpy as np
import json
from typing import Tuple, List


def create_t(num_stations: int) -> Tuple[np.array, List[int]]:
    
    """
    Creates the matrix time for a random subset of the stations.
    
    Parameters
    ----------
    num_stations : Size of the subset.
    
    Returns
    -------
    Matrix time and random idxs selected.
    """
    
    # Load the data
    with open("./prsbc/datos/202206.json") as f:
        stations = json.loads(f.readline())
    
    # Creation of the dataframe, the first one is the deposit
    longitudes = [-3.6896942]
    latitudes = [40.405611]
    for station in stations["stations"]:
        longitudes.append(station["longitude"])
        latitudes.append(station["latitude"])
    df = pd.DataFrame({"longitude": longitudes, "latitude": latitudes})
    random_idx = [0] + [np.random.randint(0, len(df))
                       for _ in range(num_stations-1)]
    df = df.loc[random_idx]
    df["id"] = range(num_stations)
    
    # Function to calculate time
    
    def calculate_time(
        lon_x : float,
        lat_x : float,
        lon_y : float,
        lat_y : float,
        speed : float = 50,
        R_t   : float = 6371
        ) -> float:
        
        """
        Calculates the time it takes to go from one station to another.
        Note that it satisfies all properties of a distance, that is,
        - d(A, B) >= 0 for all A, B.
        - d(A, B) = 0 iff A = B.
        - d(A, B) = d(B, A) for all A, B.
        - d(A, B) <= d(A, C), and d(C, B) for all A, B, C.
        where A, B and C are arbitrary locations.
        
        Parameters
        ----------
        lon_x, lat_x : Longitude and latitude of the first station.
        lon_y, lat_y : Longitude and latitude of the second station.
        speed        : Speed of the vehicle in km/h.
        R_t          : Radius of the Earth in km.
        
        Returns
        -------
        Approximated time in hours.
        """
        
        # First obtain distance in km with spherical trigonometry
        # Latitude
        delta_phi = abs(lat_x-lat_y)
        a = np.sin(delta_phi/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        lat_distance = R_t * c
        # Longitude
        delta_lambda = abs(lon_x-lon_y)
        a = np.sin(delta_lambda/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        lon_distance = R_t * c
        
        # Manhattan distance in km
        l1_distance = lat_distance + lon_distance
        
        return l1_distance * 60 / speed
    
    
    # Creation of the matrix
    t = []
    for x in range(num_stations):
        t.append([])
        for y in range(num_stations):
            coords = df[df["id"] == x][["longitude", "latitude"]].values[0]
            lon_x, lat_x = float(coords[0]), float(coords[1])
            coords = df[df["id"] == y][["longitude", "latitude"]].values[0]
            lon_y, lat_y = float(coords[0]), float(coords[1])
            t[x].append(np.int_(calculate_time(lon_x, lat_x, lon_y, lat_y)))
    
    
    return np.array(t), random_idx