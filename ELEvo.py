import datetime

import numpy as np
from scipy.stats import skewnorm


class CME:
    def __init__(self, half_width, longitude, latitude, tilt,f, initial_speed, initial_time, initial_radius, feature_type, source_of_info):

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
        self.ensemble_timesteps = None
        self.ensemble_time_resolution = None

        

def skewed_distribution(cme_speed,n_ensemble):
    speed_delta = 115
    speed_ensemble = np.random.normal(loc=cme_speed, scale=speed_delta/2, size=n_ensemble) # CME speed, normally distributed around v_0 with a standard deviation of speed_delta

    gamma_values = np.array([7.5, 22.5, 7.5, 15, 7.5, 11, 7.5, 7.5, 3.5, 3.5, 3.5, 3.5]) # Sum = 100
    gamma_values = gamma_values / np.sum(gamma_values)  # Normalize to sum to 1

    gamma_bin_edges = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4])

    gamma_bin_centers = (gamma_bin_edges[:-1] + gamma_bin_edges[1:]) / 2

    synthetic_gamma_values = np.random.choice(gamma_bin_centers, size=n_ensemble, p=gamma_values)

    gamma_shape, gamma_loc, gamma_scale = skewnorm.fit(synthetic_gamma_values)

    sampled_gamma = skewnorm.rvs(gamma_shape, loc=gamma_loc, scale=gamma_scale, size=n_ensemble)

    # remove all values below 0.05 and above 2.4
    clipped_sampled_gamma = sampled_gamma[(sampled_gamma >= 0.05) & (sampled_gamma <= 2.4)]

    # check if clipped_sampled_gamma contains enough values, if not, resample and add more values
    while len(clipped_sampled_gamma) < n_ensemble:
        additional_sample = skewnorm.rvs(gamma_shape, loc=gamma_loc, scale=gamma_scale, size=n_ensemble)
        additional_clipped_sample = additional_sample[(additional_sample >= 0.05) & (additional_sample <= 2.4)]
        clipped_sampled_gamma = np.concatenate([clipped_sampled_gamma, additional_clipped_sample])
        clipped_sampled_gamma = clipped_sampled_gamma[:n_ensemble]

    w_values = np.array([0, 0, 27.5, 26.5, 22.5, 18, 5, 0.5, 0]) # Sum = 100
    w_values = w_values / np.sum(w_values)  # Normalize to sum to 1

    w_bin_edges = np.array([120, 200, 300, 400, 500, 600, 700, 800, 900, 920])

    w_bin_centers = (w_bin_edges[:-1] + w_bin_edges[1:]) / 2

    synthetic_w_values = np.random.choice(w_bin_centers, size=n_ensemble, p=w_values)

    w_shape, w_loc, w_scale = skewnorm.fit(synthetic_w_values)

    sampled_w = skewnorm.rvs(w_shape, loc=w_loc, scale=w_scale, size=n_ensemble)

    # remove all values below 120 and above 920
    clipped_sampled_w = sampled_w[(sampled_w >= 120) & (sampled_w <= 920)]

    # check if clipped_sampled_w contains enough values, if not, resample and add more values
    while len(clipped_sampled_w) < n_ensemble:
        additional_sample = skewnorm.rvs(w_shape, loc=w_loc, scale=w_scale, size=n_ensemble)
        additional_clipped_sample = additional_sample[(additional_sample >= 120) & (additional_sample <= 920)]
        clipped_sampled_w = np.concatenate([clipped_sampled_w, additional_clipped_sample])
        clipped_sampled_w = clipped_sampled_w[:n_ensemble]

    gamma = clipped_sampled_gamma
    ambient_wind = clipped_sampled_w 

    return speed_ensemble, gamma, ambient_wind  


def create_ensemble(swinit,gammainit,cme,std_ensemble,nb_ensemble=10000,random_seed=31082026,method_type='normal'):


    np.random.seed(random_seed)


    gamma_array = None
    ambient_wind_array = None
    if method_type == 'normal':
        gamma_array        = np.random.normal(gammainit,std_ensemble['gamma'],nb_ensemble)
        ambient_wind_array = np.random.normal(swinit,std_ensemble['sw'],nb_ensemble)
        cme.initial_speed_array = np.random.normal(cme.initial_speed,std_ensemble['initial_speed'],nb_ensemble)
        cme.initial_radius_array = np.random.normal(cme.initial_radius,std_ensemble['initial_radius'],nb_ensemble)

    elif method_type=="skewed":
        cme.initial_speed_array, gamma_array, ambient_wind_array   = skewed_distribution(cme.initial_speed,nb_ensemble)
        cme.initial_radius_array = np.random.normal(cme.initial_radius,std_ensemble['initial_radius'],nb_ensemble)

    
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
    cme.ensemble_timesteps = timesteps
    cme.ensemble_time_resolution = minute_resolution

    return cme 


def calculate_ellipse_parameters(cme):
    # here we consider that all the ensembles are computed on solar wind NOT on CME parameters
    # TODO create another set of parameters were the CME direction and size are varried (but be careful not to blow up ensemble)
    theta = np.arctan(cme.f**2 * np.tan(cme.half_width))
    omega = np.sqrt(np.cos(theta)**2 * (cme.f**2 - 1) + 1)   
    cme_b = cme.cme_r_ensemble * omega * np.sin(cme.half_width) / (np.cos(cme.half_width - theta) + omega * np.sin(cme.half_width))    
    cme_a = cme_b / cme.f
    cme_c = cme.cme_r_ensemble - cme_b

    cme.cme_a = cme_a
    cme.cme_b = cme_b
    cme.cme_c = cme_c
    cme.theta = theta
    cme.omega = omega

    return cme



if __name__ == "__main__":
    cme = CME(45,0,0,0,0.7,450,datetime.datetime(2025,10,12,12,58),21.5,'SE','Dummy values')

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
    # TODO intersections = calculate_intersection(cme,spacecrafts_pos_timeframe)