from CME_class import * 
from space_object_class import *
import elevo_utils
import data_utils
import matplotlib.pyplot as plt 


def Run_one_CME(names_spcs):

    resolution = '10m'
    nb_ensemble = 500
    days_duration = 5
    time_resolution = 10

    cme = CME(45,157,-10,0,0.7,450,datetime.datetime(2025,2,10,12,58),21.5,'SE','Dummy values')
    print(cme.longitude,cme.latitude)

    start_date = "2025-02-10"
    end_date = "2025-02-15"
    spcs = []

    for name in names:                 
        data_utils.create_positions_file(name,start_date,end_date,step=resolution,save_path='data/spc_pos/',overwrite=False)
        positons_dict = data_utils.load_positions_jpl('data/spc_pos/', start_date,end_date,resolution, name)
        spcs.append(SpaceObject(name, positons_dict['lon'], positons_dict['lat'], positons_dict['r'], 'HAE', cme.initial_time, timesteps=cme.ensemble_timesteps, time_resolution=time_resolution))

    

    
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
    cme.initialize_ensemble(400,1.0e-7,std_ensemble,nb_ensemble=nb_ensemble,method_type='skewed')
    cme.propagate_cme()
    cme.calculate_ellipse_parameters()

    


    for spc in spcs:

        intersection = spc.calculate_intersection(cme)


        bound_idxs = elevo_utils.get_boundary_indices(intersection)
        arrival_times, arrival_speeds = elevo_utils.calculate_arrival(bound_idxs, cme.ensemble_timesteps, cme.cme_v_ensemble, cme.initial_time)
        debug_ensemble_idx = -1
        
        # compute how many ensemble members evetually arrive at earth
        n_arrival = np.sum(bound_idxs[0] != None)
        print(f"Number of ensemble members that arrive at {spc.name}: {n_arrival} / {bound_idxs.shape[1]} ({n_arrival/bound_idxs.shape[1]*100:.2f}%)")
        print(f"Ensemble member {debug_ensemble_idx} arrival time: {arrival_times[debug_ensemble_idx]}")
        print(f"Ensemble member {debug_ensemble_idx} arrival speed: {arrival_speeds[debug_ensemble_idx]} km/s")
        



if __name__ == "__main__":
    names = ['l1',
               'solo',
               'psp',
               'sta',
               'bepi',
               'mercury',
               'venus',
               'mars']
       
   

    Run_one_CME(names)