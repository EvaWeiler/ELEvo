import sys, os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "sc_data_functions"))

from datetime import datetime, timedelta

# PROBLEMS: Kernels still saved remotely, haven't figured out how to download from onedrive yet. 
# Aditya kernels, to download one needs login, files updated monthly.
# IMAP: data only available since September 2025 (start of mission)
#  



def get_sc_positions(sc, start_time, end_time, step, coord_sys):
    """
    Get spacecraft position(s) for one or more spacecraft.

    Parameters
    ----------
    sc : str or list of str
        Spacecraft identifier(s), e.g. 'psp', 'solo', 'sta'.
    time_series : array-like
        Times at which to compute positions.
    coord_sys : str
        Coordinate system

    Returns
    -------
    dict
        Mapping of spacecraft name -> position array.
        If a single spacecraft string was passed, returns just that array instead.
    """

    from sc_data_functions import position_frame_transforms as pft

    # Registry: maps sc name -> (module name, furnish func name, position func name, needs_coord_sys)
    sc_registry = {
        'psp':  ('functions_psp',  'psp_furnish',  'get_psp_positions'),
        'solo': ('functions_solo', 'solo_furnish', 'get_solo_positions'),
        'sta':  ('functions_sta',  'stereoa_furnish',  'get_sta_positions'),
        'stb':  ('functions_stb',  'stereob_furnish',  'get_stb_positions'),
        'imap': ('functions_imap', 'imap_furnish',  'get_imap_positions'),
        'aditya': ('functions_aditya', 'aditya_furnish',  'get_aditya_positions'),
        'juice': ('functions_juice', 'juice_furnish',  'get_juice_positions'),
        'juno': ('functions_juno', 'juno_furnish',  'get_juno_positions'),
        'L1': ('functions_lagrange', 'lagrange_furnish',  'get_lagrange_positions'),
        'L2': ('functions_lagrange', 'lagrange_furnish',  'get_lagrange_positions'),
        'L3': ('functions_lagrange', 'lagrange_furnish',  'get_lagrange_positions'),
        'L4': ('functions_lagrange', 'lagrange_furnish',  'get_lagrange_positions'),
        'L5': ('functions_lagrange', 'lagrange_furnish',  'get_lagrange_positions'),
        'uly': ('functions_ulysses', 'ulysses_furnish',  'get_ulysses_positions'),
        'wind': ('functions_wind', 'download_wind_orb', 'get_wind_positions'),
        'ace': ('functions_ace', 'download_ace_mag', 'get_acepos_frommag_range')
    }

    time_series = [start_time]
    time = start_time
    while time <= end_time:
        time += timedelta(minutes=step)
        time_series.append(time)

    # Allow single string or list input
    sc_list = [sc] if isinstance(sc, str) else sc

    results = {}
    for name in sc_list:
        if name not in sc_registry:
            raise ValueError(f"Unknown spacecraft '{name}'. Available: {list(sc_registry.keys())}")

        module_name, furnish_name, pos_func_name = sc_registry[name]

        module = __import__(f'sc_data_functions.{module_name}', fromlist=[module_name])

        pos_func = getattr(module, pos_func_name)

        if name == 'wind':

            wind_dir = os.path.dirname('../kernels/wind/')

            if not os.path.exists(wind_dir):   
                os.makedirs(wind_dir)

            getattr(module, furnish_name)(start_time, end_time, path=wind_dir+'/')

            print(wind_dir)
            pos = pos_func(start_time, end_time, coord_sys='HAE', path=wind_dir+'/orbit/')


        elif name == 'ace':

            data_dir = os.path.dirname('../kernels/')
            ace_dir = os.path.dirname('../kernels/ace/')

            if not os.path.exists(data_dir):
                os.makedirs(data_dir)
            if not os.path.exists(ace_dir):   
                os.makedirs(ace_dir)
        
            getattr(module, furnish_name)(start_time, end_time, path=ace_dir+'/')

            ace_pos = pos_func(start_time, end_time, path=ace_dir)
            #print('ACE time resolution: ', (ace_pos.time[1]-ace_pos.time[0]).total_seconds()/60.)
            ace_pos = ace_pos.set_index('time')
            ace_pos = (ace_pos.resample(str(int(step))+'min').mean().interpolate(method='time')).reset_index()

            if coord_sys == 'GSE':
                pos_conv = ace_pos

            if coord_sys == 'GSM':
                pos_conv = pft.GSE_to_GSM(ace_pos)

            if coord_sys in ['HAE', 'ECLIPJ2000']:
                pos_conv = pft.GSE_to_HAE(ace_pos)

            if coord_sys == 'HEEQ':
                pos_hae = pft.GSE_to_HAE(ace_pos)
                pos_conv = pft.HAE_to_HEEQ(pos_hae)

            pos = pos_conv
            

        else:
            getattr(module, furnish_name)()  # furnish

            if name in ['L1', 'L2', 'L3', 'L4', 'L5']:
                pos = pos_func(time_series=time_series, lagrange_point=name, coord_sys=coord_sys)
            else:
                pos = pos_func(time_series=time_series, coord_sys=coord_sys)

        results[name] = pos

    # Keep backward-compatible behavior: single sc string -> return array directly
    return results[sc] if isinstance(sc, str) else results


def get_planets_positions(planets, time_series, coord_sys):
    """
    Get planet position(s) for one or more planet.

    Parameters
    ----------
    sc : str or list of str
        Planet identifier(s), e.g. 'mercury', 'venus', 'earth'.
    time_series : array-like
        Times at which to compute positions.
    coord_sys : str
        Coordinate system

    Returns
    -------
    dict
        Mapping of planet name -> position array.
        If a single planet string was passed, returns just that array instead.
    """

    from sc_data_functions import functions_planets as fplanets

    planet_registry = {
        'mercury barycenter': (1),  
        'saturn barycenter': (6),
        'earth barycenter': (3),
        'neptune barycenter': (8),
        'venus barycenter': (2),
        'uranus barycenter': (7), 
        'mars barycenter': (4),     
        'pluto barycenter': (9),
        'mercury': (199),
        'venus': (299),
        'moon': (301),
        'earth': (399),
        'jupiter barycenter': (5),
        'sun': (10)
    }
    
    planets_list = [planets] if isinstance(planets, str) else planets

    fplanets.generic_furnish()

    results = {}
    for name in planets_list:
        if name not in planet_registry:
            raise ValueError(f"Unknown planet '{name}'. Available: {list(planet_registry.keys())}")

        pos = fplanets.get_planet_positions(time_series=time_series,planet=name,coord_sys=coord_sys)

        results[name] = pos
    
    # Keep backward-compatible behavior: single sc string -> return array directly
    return results[planets] if isinstance(planets, str) else results


if __name__ == "__main__":
    sc_list = ['psp', 'solo', 'ace', 'wind', 'L1', 'L5', 'stb', 'sta', 'uly', 'juno', 'juice', 'imap', 'aditya']
    planets_list = ['mercury','venus','earth','mars barycenter','jupiter barycenter','saturn barycenter','uranus barycenter','neptune barycenter'] 
    coord_system = 'ECLIPJ2000'

    #pos_planet = get_planets_positions(planets=planets_list, time_series=time_in_min_res, coord_sys=coord_system)
    pos_sc = get_sc_positions(sc=sc_list, start_time=datetime(2026,1,1), end_time=datetime(2026,1,5), step=10., coord_sys=coord_system)

    #print(pos_planet)
    print(pos_sc)