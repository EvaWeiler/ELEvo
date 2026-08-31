import numpy as np 
from datetime import datetime

class CME:
    def __init__(self, half_width, longitude, latitude, tilt,f, initial_speed, initial_time, initial_radius, feature_type,source_of_info):

        ## These are the initial parameters for this CME, their values can be set and changed depending on source and case scenario
        self.half_width = half_width
        self.longitude = longitude
        self.latitude = latitude
        self.tilt = tilt
        self.f = f
        self.initial_speed = initial_speed
        self.initial_time = initial_time
        self.initial_radius = initial_radius
        self.feature_type = feature_type
        self.source_of_info = source_of_info


        # These are the ensemble values resulting of meshgrid of all  ensembles. They are used for computations 
        self.half_width_array = None
        self.longitude_array = None
        self.latitude_array = None
        self.tilt_array = None
        self.initial_speed_array = None
        self.initial_array = None

        # result radial distance and velocity of CME apex for all timesteps.
        self.cme_r_ensemble = None
        self.cme_v_ensemble = None

        
           


def create_ensemble(swinit,gammainit,cme,nb_ensembles,random_seed=31082026,method_type='normal'):


    np.random.seed(random_seed)


    gamma_array = None
    ambient_wind_array = None
    if method_type == 'normal':
        for k in nb_ensembles.keys():
            if k =='solarwind':
                gamma_array        = np.random.normal(gammainit,0.025,nb_ensembles[k]['gamma'])
                ambient_wind_array = np.random.normal(swinit,50,nb_ensembles[k]['sw'])
            else:
                cme.initial_speed_array = np.random.normal(cme.initial_speed,1,nb_ensembles[k]['initial_speed'])
                cme.initial_radius_array = np.random.normal(cme.initial_radius,1,nb_ensembles[k]['initial_radius'])

    combinations = np.meshgrid( gamma_array,
                                ambient_wind_array,
                                cme.initial_speed_array,
                                cme.initial_radius_array)

    ensemble_combinations = np.array(combinations).T.reshape(-1, 8)
    print('There are ',ensemble_combinations.shape[0],' ensemble members')


    gamma_array= ensemble_combinations[:,0]
    ambient_wind_array= ensemble_combinations[:,1]
    cme.initial_speed_array= ensemble_combinations[:,2]
    cme.initial_radius_array= ensemble_combinations[:,3]

    return gamma_array, ambient_wind_array, cme


def propagate_cme(gamma_array, ambient_wind_array,cme,days_duration=5,minute_resolution=10):


    timesteps = np.arange(days_duration*minute_resolution*60.0)
    timesteps = np.repeat(timesteps[None,:],repeats=cme.initial_radius_array.shape[0],axis=0)
    timesteps = np.transpose(timesteps)

    distance0_list = cme.initial_radius_array
    accsign = np.ones(distance0_list.shape)
    accsign[cme.initial_speed_array < ambient_wind_array] = -1.
    
    cme_r_ensemble = (accsign / (gamma_array * 1e-7)) * np.log(1 + (accsign * (gamma_array * 1e-7) * ((cme.initial_speed_array - ambient_wind_array) * timesteps))) + ambient_wind_array * timesteps + distance0_list
    cme_v_ensemble = (cme.initial_speed_array - ambient_wind_array) / (1 + (accsign * (gamma_array * 1e-7) * (cme.initial_speed_array - ambient_wind_array) * timesteps)) + ambient_wind_array


    cme.cme_r_ensemble = cme_r_ensemble
    cme.cme_v_ensemble = cme_v_ensemble

    return cme 


def calculate_ellipse_parameters(cme):
    # here we consider that all the ensembles are computed on solar wind NOT on CME parameters
    # TODO create another set of parameters were the CME direction and size are varried (but be careful not to blow up ensemble)
    theta = np.arctan(cme.f**2 * np.tan(cme.half_width))
    omega = np.sqrt(np.cos(theta)**2 * (cme.f**2 - 1) + 1)   
    cme_b = cme.cme_r_ensemble * omega * np.sin(cme.half_width) / (np.cos(cme.half_width - theta) + omega * np.sin(cme.half_width))    
    cme_a = cme_b / cme.f
    cme_c = cme.cme_r_ensemble - cme_b

    print(theta,omega,cme_b.shape,cme_a.shape,cme_c.shape)



if __name__ == "__main__":
    cme = CME(45,0,0,0,0.7,450,datetime(2025,10,12,12,58),21.5,'SE','Dummy values')

    nb_ensembles = {
            'solarwind':{
                'sw': 20,
                'gamma': 20,
                
            },
            'cme' : {
                'half_width' : 1, # this could be change to add more ensemble when reconstructing the CMEs /checking intersection
                'longitude' : 1,# this could be change to add more ensemble when reconstructing the CMEs /checking intersection
                'latitude' : 1,# this could be change to add more ensemble when reconstructing the CMEs /checking intersection
                'tilt' : 1,# this could be change to add more ensemble when reconstructing the CMEs /checking intersection
                'initial_speed' : 10,
                # 'initial_time' : 10, # I havent added that because would need to calculate ensemble for time, which is not straightforward 
                'initial_radius' : 10,
            }
    }
    gamma_array, ambient_wind_array, cme = create_ensemble(400,1.0e-7,cme,nb_ensembles)
    cme = propagate_cme(gamma_array, ambient_wind_array,cme,days_duration=5,minute_resolution=10)
    cme = calculate_ellipse_parameters(cme)
    # TODO intersections = calculate_intersection(cme,spacecrafts_pos_timeframe)