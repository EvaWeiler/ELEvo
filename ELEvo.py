from CME_class import * 
from Space_object_class import *
import elevo_utils

def Run_one_CME():
    cme = CME(45,0,0,0,0.7,450,datetime.datetime(2025,10,12,12,58),21.5,'SE','Dummy values')
    std_ensemble = {
            'sw': 50,
            'gamma': 0.025,
            'half_width' : 0, # this could be change to add more ensemble when reconstructing the CMEs /checking intersection
            'longitude' : np.deg2rad(0),# this could be change to add more ensemble when reconstructing the CMEs /checking intersection
            'latitude' : np.deg2rad(0),# this could be change to add more ensemble when reconstructing the CMEs /checking intersection
            'tilt' : 0,# this could be change to add more ensemble when reconstructing the CMEs /checking intersection
            'initial_speed' : 100,
            'initial_radius' : 2,
                
        }
    cme.initialize_ensemble(400,1.0e-7,std_ensemble,nb_ensemble=20000,method_type='skewed')
    cme.propagate_cme()
    cme.calculate_ellipse_parameters()


    earth_r = np.zeros(np.shape(cme.cme_r_ensemble)[0])
    earth_r[:] = 150*10**6

    earth_lon = np.linspace(0,45.0,np.shape(cme.cme_r_ensemble)[0])
    earth_lon = np.deg2rad(earth_lon)
    earth_lat = np.zeros(np.shape(cme.cme_r_ensemble)[0])
    earth_timegrid = cme.ensemble_timesteps
    earth_time_resolution = cme.ensemble_time_resolution
    
    earth_object = SpaceObject('Earth', earth_lon, earth_lat, earth_r, 'HAE', cme.initial_time, timesteps=earth_timegrid, time_resolution=earth_time_resolution)

    intersection = earth_object.calculate_intersection(cme)


    bound_idxs = elevo_utils.get_boundary_indices(intersection)
    arrival_times, arrival_speeds = elevo_utils.calculate_arrival(bound_idxs, cme.ensemble_timesteps, cme.cme_v_ensemble, cme.initial_time)
    debug_ensemble_idx = -1
    
    # compute how many ensemble members evetually arrive at earth
    n_arrival = np.sum(bound_idxs[0] != None)
    print(f"Number of ensemble members that arrive at Earth: {n_arrival} / {bound_idxs.shape[1]} ({n_arrival/bound_idxs.shape[1]*100:.2f}%)")
    print(f"Ensemble member {debug_ensemble_idx} arrival time: {arrival_times[debug_ensemble_idx]}")
    print(f"Ensemble member {debug_ensemble_idx} arrival speed: {arrival_speeds[debug_ensemble_idx]} km/s")
        



if __name__ == "__main__":
   Run_one_CME()