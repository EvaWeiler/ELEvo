from CME_class import * 
from space_object_class import *
import elevo_utils
import data_utils
import plot_utils


def Run_one_CME(names,resolution=10,nb_ensemble=10000,days_duration=5):

    cme = CME(half_width=np.deg2rad(45),
              longitude=np.deg2rad(157),
              latitude=np.deg2rad(-10),
              tilt=0,
              f=0.7,
              initial_speed=450,
              initial_time=datetime.datetime(2025,2,12,12,58),
              initial_radius=21.5,
              ensemble_time_resolution=resolution,
              nb_ensemble=nb_ensemble,
              days_duration=days_duration,
              feature_type='SE',
              source_of_info='Dummy values')




    start_date = cme.time_array[0].strftime("%Y-%m-%d") 
    end_date = cme.time_array[-1].strftime("%Y-%m-%d") 

    spcs = data_utils.get_spcs_dates(start_date,end_date,names,str(resolution)+'m',cme.ensemble_timesteps)

    
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
    cme.initialize_ensemble(400,1.0e-7,std_ensemble,method_type='skewed')
    cme.propagate_cme()
    cme.calculate_ellipse_parameters()

    
    intersections = {}
    for spc in spcs:

        intersection = spc.calculate_intersection(cme)
        bound_idxs = elevo_utils.get_boundary_indices(intersection)
        arrival_times, arrival_speeds = elevo_utils.calculate_arrival(bound_idxs, cme.ensemble_timesteps, cme.cme_v_ensemble, cme.initial_time)

        intersections[spc.name] = {
            'intersecion':intersection,
            'bound_idxs':bound_idxs,
            'arrival_times':arrival_times,
            'arrival_speeds':arrival_speeds,
        }
        
    return cme, spcs,intersections


def print_arrivals(intersections):
    for spc in intersections.keys():
        bound_idxs = intersections[spc]['bound_idxs']
        arrival_times = intersections[spc]['arrival_times']
        clean_datetime = arrival_times[arrival_times != None].astype('datetime64[ns]')
        arrival_speeds = intersections[spc]['arrival_speeds']
        n_arrival = np.sum(bound_idxs[0] != None)
        print(f"Number of ensemble members that arrive at {spc}: {n_arrival} / {bound_idxs.shape[1]} ({n_arrival/bound_idxs.shape[1]*100:.2f}%)")

        if n_arrival>0:
            mean_time   = np.mean(clean_datetime.view('i8')).astype('datetime64[ns]')
            mean_time_str = np.datetime_as_string(mean_time, unit='s').split('T')[-1]

            
            std_time = np.std(clean_datetime.view('i8')).astype('timedelta64[ns]')
            std_hours = std_time / np.timedelta64(1, 'h')

            mean_speeds = np.mean(arrival_speeds[arrival_speeds != None])
            std_speeds = np.std(arrival_speeds[arrival_speeds != None])

            print(f"Average ensemble arrival time: {mean_time_str} ± {std_hours:.2f} hours")
            print(f"Average  ensemble arrival speed: {mean_speeds} \u00B1 {std_speeds}")


if __name__ == "__main__":
    names = ['l1',
               'solo',
               'psp',
               'sta',
               'bepi',
               'mercury',
               'venus',
               'mars']
       
   

    cme, spcs,intersections = Run_one_CME(names)

    # print_arrivals(intersections)

    plot_utils.make_frame_sequence("2025-02-12", "2025-02-17",cme,resolution=10)