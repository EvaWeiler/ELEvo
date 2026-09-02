import datetime

import numpy as np

from ELEvo import CME, calculate_ellipse_parameters, create_ensemble, propagate_cme


class SpaceObject:
    def __init__(self, name, longitude, latitude, radial_distance, coordinate_system, timesteps, time_resolution):
        self.name = name
        self.longitude = longitude  
        self.latitude = latitude     
        self.radial_distance = radial_distance
        self.coordinate_system = coordinate_system

        self.timesteps = timesteps
        self.time_resolution = time_resolution

def calculate_intersection(cme,spacecrafts_pos_timeframe):
    pass

if __name__ == "__main__":

    initial_time = datetime.datetime(2025,10,12,12,58)

    cme = CME(45,0,0,0,0.7,450,initial_time,21.5,'SE','Dummy values')

    std_ensemble = {
        'sw': 50,
        'gamma': 0.025,
        'half_width' : 0, # this could be change to add more ensemble when reconstructing the CMEs /checking intersection
        'longitude' : 0,# this could be change to add more ensemble when reconstructing the CMEs /checking intersection
        'latitude' : 0,# this could be change to add more ensemble when reconstructing the CMEs /checking intersection
        'tilt' : 0,# this could be change to add more ensemble when reconstructing the CMEs /checking intersection
        'initial_speed' : 100,
        # 'initial_time' : 10, # I havent added that because would need to calculate ensemble for time, which is not straightforward 
        'initial_radius' : 2,
            
    }
    gamma_array, ambient_wind_array, cme = create_ensemble(400,1.0e-7,cme,std_ensemble,nb_ensemble=20000,method_type='skewed')
    cme = propagate_cme(gamma_array, ambient_wind_array,cme,days_duration=5,minute_resolution=10)
    cme = calculate_ellipse_parameters(cme)

    earth_r = np.zeros(np.shape(cme.cme_r_ensemble)[0])
    earth_r[:] = 1.0
    earth_lon = np.zeros(np.shape(cme.cme_r_ensemble)[0])
    earth_lat = np.zeros(np.shape(cme.cme_r_ensemble)[0])
    earth_timegrid = cme.ensemble_timesteps
    earth_time_resolution = cme.ensemble_time_resolution

    earth = SpaceObject('Earth', earth_lon, earth_lat, earth_r, 'HAE', timegrid=earth_timegrid, time_resolution=earth_time_resolution)
    # TODO intersections = calculate_intersection(cme,spacecrafts_pos_timeframe)