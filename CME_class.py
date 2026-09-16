from dataclasses import dataclass
import numpy as np 
import datetime
import elevo_utils
import astropy.units as u
from datetime import datetime,timedelta

@dataclass
class CME:

    half_width: float
    longitude: float
    latitude: float
    tilt: float
    f: float 
    initial_speed: float 
    initial_time : datetime
    initial_radius :float 
    ensemble_time_resolution : float
    nb_ensemble:int
    days_duration: int
    feature_type : str
    source_of_info : str

    half_width_array :  np.array = None
    longitude_array : np.array = None
    latitude_array : np.array = None
    tilt_array : np.array = None
    initial_speed_array : np.array = None
    initial_array : np.array = None
    gamma_array: np.array = None
    ambient_wind_array : np.array = None

    cme_r_ensemble : np.array = None
    cme_v_ensemble : np.array = None
    ensemble_timesteps : np.array = None
    time_array: np.array = None

    def __post_init__(self):
        self.ensemble_timesteps = elevo_utils.create_time_grid(self.nb_ensemble,self.days_duration,self.ensemble_time_resolution)
        self.time_array  = np.arange(self.initial_time, 
                                     self.initial_time + timedelta(days=self.days_duration), 
                                     timedelta(minutes=self.ensemble_time_resolution)).astype(datetime)

    def initialize_ensemble(self, swinit,gammainit,std_ensemble,random_seed=31082026,method_type='normal'):

        np.random.seed(random_seed)
        if method_type == 'normal':
            self.gamma_array        = np.random.normal(gammainit,std_ensemble['gamma'],self.nb_ensemble)
            self.ambient_wind_array = np.random.normal(swinit,std_ensemble['sw'],self.nb_ensemble)
            self.initial_speed_array = np.random.normal(self.initial_speed,std_ensemble['initial_speed'],self.nb_ensemble)
            self.initial_radius_array = np.random.normal(self.initial_radius,std_ensemble['initial_radius'],self.nb_ensemble)

        elif method_type=="skewed":
            self.initial_speed_array, self.gamma_array, self.ambient_wind_array   = elevo_utils.skewed_distribution(self.initial_speed,self.nb_ensemble)
            self.initial_radius_array = np.random.normal(self.initial_radius,std_ensemble['initial_radius'],self.nb_ensemble)

        
        
    def propagate_cme(self):

        distance0_list = self.initial_radius_array*u.R_sun.to(u.km) 
        accsign = np.ones(distance0_list.shape)
        accsign[self.initial_speed_array < self.ambient_wind_array] = -1.
        
        cme_r_ensemble = (accsign / (self.gamma_array * 1e-7)) * np.log(1 + (accsign * (self.gamma_array * 1e-7) * ((self.initial_speed_array - self.ambient_wind_array) * self.ensemble_timesteps))) + self.ambient_wind_array * self.ensemble_timesteps + distance0_list
        cme_v_ensemble = (self.initial_speed_array - self.ambient_wind_array) / (1 + (accsign * (self.gamma_array * 1e-7) * (self.initial_speed_array - self.ambient_wind_array) * self.ensemble_timesteps)) + self.ambient_wind_array

        self.cme_r_ensemble = cme_r_ensemble
        self.cme_v_ensemble = cme_v_ensemble


    def calculate_ellipse_parameters(self):
        # here we consider that all the ensembles are computed on solar wind NOT on CME parameters
        # TODO create another set of parameters were the CME direction and size are varried (but be careful not to blow up ensemble)
        theta = np.arctan(self.f**2 * np.tan(self.half_width))
        omega = np.sqrt(np.cos(theta)**2 * (self.f**2 - 1) + 1)   
        cme_b = self.cme_r_ensemble * omega * np.sin(self.half_width) / (np.cos(self.half_width - theta) + omega * np.sin(self.half_width))    
        cme_a = cme_b / self.f
        cme_c = self.cme_r_ensemble - cme_b

        self.cme_a = cme_a
        self.cme_b = cme_b
        self.cme_c = cme_c
        self.theta = theta
        self.omega = omega


    def cme_orthonormal_base(self):
        """Compute orthonormal base for CME coordinates.

        Parameters
        ----------
        cme : CME
            The CME object containing the longitude and latitude.

        Returns
        -------
        array-like
            Set of vectors spanning an orthonormal base for the CME. Dimension is (3,3).
        """

        lon, lat = self.longitude, self.latitude

        # Radial direction from the Sun towards the CME apex
        u_r = np.array([np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)])
        # Vector in the ecliptic plane, perpendicular to the CME's longitude direction
        e_lon = np.array([-np.sin(lon), np.cos(lon), 0.0])
        # Vector perpendicular to both, tilted out of the ecliptic plane by the CME's latitude
        e_lat = np.array([-np.sin(lat)*np.cos(lon), -np.sin(lat)*np.sin(lon), np.cos(lat)])

        return e_lon, u_r, e_lat
            
    def plot_ellipse_at_time(self, target_time, ax=None, n_points=200,
                          unit=u.AU, ensemble_alpha=0.15,
                          ensemble_color='steelblue', show_mean=True):
        """
        Plot the ensemble of CME ellipses in the ecliptic plane at the
        timestep closest to `target_time` (a datetime object).
        """
        # 1. find nearest index in time_array (array of python datetimes)
        diffs = np.array([abs((t - target_time).total_seconds())
                        for t in self.time_array])
        i_t = np.argmin(diffs)

        if diffs[i_t]/60.0>10.0:
            return


        a = self.cme_a[i_t,:]
        b = self.cme_b[i_t,:]
        c = self.cme_c[i_t,:]

      

        t = ((np.arange(201)-10)*np.pi/180)-(self.longitude)
        t1 = ((np.arange(201)-10)*np.pi/180)


        xs = (c*np.cos(self.longitude))[:,None]+((a*b)[:,None]/np.sqrt((b[:,None]*np.cos(t1))**2+(a[:,None]*np.sin(t1))**2))*np.sin(t)
        ys = (c*np.sin(self.longitude))[:,None]+((a*b)[:,None]/np.sqrt((b[:,None]*np.cos(t1))**2+(a[:,None]*np.sin(t1))**2))*np.cos(t)


        theta = np.arctan2(ys, xs)
        r=np.sqrt(xs**2+ys**2)
        

        for i in range(r.shape[0]):
            ax.plot(theta[i], r[i], color=ensemble_color,
                    alpha=ensemble_alpha, lw=0.8)

        if show_mean:
            ax.plot(theta.mean(axis=0), r.mean(axis=0),
                    color='crimson', lw=2, label='ensemble mean')

        return ax
    



if __name__ == "__main__":
    cme = CME(45,0,0,0,0.7,450,datetime.datetime(2025,10,12,12,58),21.5,'SE','Dummy values')
    std_ensemble = {
            'sw': 50,
            'gamma': 0.025,
            'half_width' : 0, # this could be change to add more ensemble when reconstructing the CMEs /checking intersection
            'longitude' : np.deg2rad(0),# this could be change to add more ensemble when reconstructing the CMEs /checking intersection
            'latitude' : np.deg2rad(0),# this could be change to add more ensemble when reconstructing the CMEs /checking intersection
            'tilt' : 0,# this could be change to add more ensemble when reconstructing the CMEs /checking intersection
            'initial_speed' : 100,
            # 'initial_time' : 10, # I havent added that because would need to calculate ensemble for time, which is not straightforward 
            'initial_radius' : 2,
                
        }
    cme.initialize_ensemble(400,1.0e-7,std_ensemble,nb_ensemble=20000,method_type='skewed')
    cme.propagate_cme()
    cme.calculate_ellipse_parameters()
