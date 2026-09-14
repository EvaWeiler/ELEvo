import sys, os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "sc_data_functions"))

from datetime import datetime, timedelta

# do not know how to handle different spacecraft at L1, 
# differently coded in comparison to other spacecraft functions in Emmas github repo


def get_sc_positions(sc, time_series, coord_sys):
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

    # Registry: maps sc name -> (module name, furnish func name, position func name, needs_coord_sys)
    sc_registry = {
        'psp':  ('functions_psp',  'psp_furnish',  'get_psp_positions'),
        'solo': ('functions_solo', 'solo_furnish', 'get_solo_positions'),
        'sta':  ('functions_sta',  'stereoa_furnish',  'get_sta_positions'),
        'stb':  ('functions_stb',  'stereob_furnish',  'get_stb_positions'),
        'imap': ('functions_imap', 'imap_furnish',  'get_imap_positions'),
        'juice': ('functions_juice', 'juice_furnish',  'get_juice_positions'),
        'juno': ('functions_juno', 'juno_furnish',  'get_juno_positions'),
        'L1': ('functions_lagrange', 'lagrange_furnish',  'get_lagrange_positions'),
        'uly': ('functions_ulysses', 'ulysses_furnish',  'get_ulysses_positions'),
        'vex': ('functions_vex', 'vex_furnish',  'get_vex_positions')
    }

    # Allow single string or list input
    sc_list = [sc] if isinstance(sc, str) else sc

    results = {}
    for name in sc_list:
        if name not in sc_registry:
            raise ValueError(f"Unknown spacecraft '{name}'. Available: {list(sc_registry.keys())}")

        module_name, furnish_name, pos_func_name = sc_registry[name]

        module = __import__(f'sc_data_functions.{module_name}', fromlist=[module_name])
        getattr(module, furnish_name)()  # furnish

        pos_func = getattr(module, pos_func_name)

        if name == 'L1':
            pos = pos_func('L1', time_series=time_series, coord_sys=coord_sys)
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
    sc_list = ['solo', 'psp', 'sta']
    planets_list = ['mercury','venus','earth','mars barycenter','jupiter barycenter','saturn barycenter','uranus barycenter','neptune barycenter'] 
    coord_system = 'ECLIPJ2000'
    res_in_min = 10.
    time_in_min_res = [datetime(2026,1,19,0)+timedelta(minutes=res_in_min*n) for n in range(100)]

    pos_planet = get_planets_positions(planets=planets_list, time_series=time_in_min_res, coord_sys=coord_system)
    pos_sc = get_sc_positions(sc=sc_list, time_series=time_in_min_res, coord_sys=coord_system)

    print(pos_planet)
    print(pos_sc)