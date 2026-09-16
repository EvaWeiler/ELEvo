import os 
import datetime
import requests
import json 
import ELEvo
import numpy as np 
import astropy.units as u
from astropy.coordinates import SkyCoord, HeliocentricMeanEcliptic
from sunpy.coordinates import get_horizons_coord,frames
from astropy.time import Time
import sunpy.coordinates  
from astropy import constants as const
from datetime import datetime,timedelta
import elevo_utils
import pickle
import re
from Space_object_class import SpaceObject


def sphere_to_cart_heeq(r, lat_rad, lon_rad):
    x = r * np.cos(lat_rad) * np.cos(lon_rad)
    y = r * np.cos(lat_rad) * np.sin(lon_rad)
    z = r * np.sin(lat_rad)

    return x, y, z

def cart_heeq_to_sphere(x,y,z):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arctan2(y,x)
    phi = np.arctan2(z,np.sqrt(x**2 + y**2))
    return r,phi,theta


def convert_HEEQ_to_HAE(long,lat,radius,time_obs):
    r_sun_in_au = const.R_sun.to(u.au)
    au_in_rsun = const.au.to(u.R_sun)
    long_radian = np.rad2deg(long)
    lat_radian  = np.rad2deg(lat)
    x,y,z = sphere_to_cart_heeq(radius, lat_radian, long_radian)  
    obstime = Time(time_obs.replace('T', ' '))
    heeq_coord = SkyCoord(
        x=x*r_sun_in_au,
        y=y*r_sun_in_au,
        z=z*r_sun_in_au, 
        representation_type='cartesian', 
        frame='heliographic_stonyhurst', 
        obstime=obstime
    )
    hae_coord = heeq_coord.transform_to(HeliocentricMeanEcliptic(obstime=obstime))
    hae_cartesian = hae_coord.cartesian
    r,phi,theta = cart_heeq_to_sphere(hae_cartesian.x*au_in_rsun,hae_cartesian.y*au_in_rsun,hae_cartesian.z*au_in_rsun)
    return np.rad2deg(theta.value),np.rad2deg(phi.value),r.value

def load_donki_cmes(path_json,resolution,nb_ensemble,days_duration):
    # Opening JSON file
    with open(path_json) as json_file:
        donki_data = json.load(json_file)

    CMEs = []
    for index, row in enumerate(donki_data):
        for analysis in row['cmeAnalyses']:
            if analysis['longitude'] is not None and analysis['latitude'] is not None:
                long,lat,radius = convert_HEEQ_to_HAE(analysis['longitude'],analysis['latitude'],21.5,analysis['time21_5'])
                cme = ELEvo.CME(np.deg2rad(analysis['halfAngle']),
                                np.deg2rad(long),
                                np.deg2rad(lat),
                                np.deg2rad(0.0),
                                0.7,
                                analysis['speed'],
                                analysis['time21_5'],
                                21.5,
                                ensemble_time_resolution=resolution,
                                nb_ensemble=nb_ensemble,
                                days_duration=days_duration,
                                feature_type=analysis['featureCode'],
                                source_of_info='DONKI'
                                )
                CMEs.append(cme)
    return CMEs


def download_donki_cmes(date_start,date_end,data_type='CME',file_path='data/'):
    """
    Download DONKI data for the specified data type and save it as a JSON file.

    """

    data_dir = os.path.dirname(file_path)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    url_donki = 'https://kauai.ccmc.gsfc.nasa.gov/DONKI/WS/get/'+data_type+'?startDate='+date_start.strftime("%Y-%m-%d")+'&endDate='+date_end.strftime("%Y-%m-%d")

    try:
        response = requests.get(url_donki, timeout=600)
        response.raise_for_status()  # Raises HTTPError for bad responses (4xx/5xx)

        # Save content as JSON file
        with open(f"{file_path}{date_start.strftime("%Y-%m-%d")}_{date_end.strftime("%Y-%m-%d")}_HAE.json", 'w', encoding='utf-8') as f:
            f.write(response.text)

    except requests.RequestException as e:
        raise RuntimeError(f"Failed to download DONKI data: {e}")

def parse_space_obj_names(name):

    name_clean = name.lower()
    name_clean = re.sub(r'[^a-z0-9]', '', name_clean)

    name_num_mappings = {
        -8: ['wind'],
        -96: ['psp', 'parker', 'parkersolarprobe', 'spp', 'solarprobeplus'],
        -121: ['bepi', 'bepicolombo'],
        -144: ['solo', 'solarorbiter'],
        -234: ['sta', 'stereoa', 'stereoahead'],
        -235: ['stb', 'stereob', 'stereobehind'],
        31: ['l1', 'sembl1'],
        199: ['mercury'],
        299: ['venus'],
        399: ['earth'],
        499: ['mars'],
        599: ['jupiter'],
        699: ['saturn'],
        799: ['uranus'],
        899: ['neptune'],
    }

    name_mapping ={
        -8: 'wind',
        -96: 'psp',
        -121: 'bepi',
        -144: 'solo',
        -234: 'sta',
        -235: 'stb',
        31: 'l1',
        199: 'mercury',
        299: 'venus',
        399: 'earth',
        499: 'mars',
        599: 'jupiter',
        699: 'saturn',
        799: 'uranus',
        899: 'neptune',
    }

    name_mapped = [code for code, names in name_num_mappings.items() if name_clean in names]

    if len(name_mapped) == 1:
        return name_mapped[0], name_mapping[name_mapped[0]]
    else:
        raise ValueError(f"Unknown or ambiguous space object name: {name}")
    
def create_positions_file(space_obj,start,stop,step='10min',save_path='',overwrite=False):

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    name = space_obj+"_"+step+"_"+start+"_"+stop+'_HAE.pkl'

    if os.path.exists(save_path+name) and overwrite==False:
        print('The coordinate file is already there not downloading...')
        return 

    start_dt = datetime.strptime(start, '%Y-%m-%d')
    stop_dt = datetime.strptime(stop, '%Y-%m-%d')

    num_seconds = (stop_dt - start_dt).total_seconds()
    step_size_seconds = int(step.split('m')[0])*60

    max_lines_jpl = 90024

    coord_dict = {}

    space_obj_code, space_obj_name = parse_space_obj_names(space_obj)

    if num_seconds/step_size_seconds > max_lines_jpl:
        num_exceed = np.ceil((num_seconds/step_size_seconds)/max_lines_jpl)
        print(f"Number of lines exceeds JPL Horizons maximum of {max_lines_jpl} by a factor of {num_exceed:.2f}.")

        obj_time = []
        obj_r = []
        obj_lon = []
        obj_lat = []
        obj_x = []
        obj_y = []
        obj_z = []
        
        # Split into multiple requests
        current_start = start_dt

        while current_start < stop_dt:
            current_stop = current_start + timedelta(seconds=(max_lines_jpl-1)*step_size_seconds)
            if current_stop > stop_dt:
                current_stop = stop_dt

            timerange = {'start':current_start, 'stop':current_stop, 'step':step}
            try:
                coord = get_horizons_coord(space_obj_code, timerange)
                coord = coord.transform_to(HeliocentricMeanEcliptic(obstime=coord.obstime))

            except ValueError as e:
                print(f"Error retrieving data for {space_obj} from {current_start} to {current_stop}: {e}")
                calc_times = [current_start + timedelta(seconds=i*step_size_seconds) for i in range(int((current_stop - current_start).total_seconds()/step_size_seconds + 1))]
                calc_times = [datetime.strptime(calc_times[i].strftime('%Y-%m-%d %H:%M'+':00'), '%Y-%m-%d %H:%M:%S') for i in range(len(calc_times))]

                obj_time.extend(calc_times)
                obj_r.extend([np.nan]*int((current_stop - current_start).total_seconds()/step_size_seconds + 1))
                obj_lon.extend([np.nan]*int((current_stop - current_start).total_seconds()/step_size_seconds + 1))
                obj_lat.extend([np.nan]*int((current_stop - current_start).total_seconds()/step_size_seconds + 1))

                obj_x.extend([np.nan]*int((current_stop - current_start).total_seconds()/step_size_seconds + 1))
                obj_y.extend([np.nan]*int((current_stop - current_start).total_seconds()/step_size_seconds + 1))
                obj_z.extend([np.nan]*int((current_stop - current_start).total_seconds()/step_size_seconds + 1))

                current_start = current_stop + timedelta(seconds=step_size_seconds)
                continue

          
            # remove leap seconds because datetime is not comaptible with them
            obj_time.extend([datetime.strptime(coord[i].obstime.strftime('%Y-%m-%d %H:%M'+':00'), '%Y-%m-%d %H:%M:%S') for i in range(len(coord))])#obj_time.extend(coord.obstime.to_datetime())
            obj_r.extend(coord.distance.value * u.au.to(u.km) )
            obj_lon.extend(np.deg2rad(coord.lon.value))
            obj_lat.extend(np.deg2rad(coord.lat.value))

            x,y,z = elevo_utils.convert_to_cartesian(np.deg2rad(np.array(coord.lon.value)),np.deg2rad(np.array(coord.lat.value)), np.array(coord.distance.value * u.au.to(u.km) ) )
            obj_x.extend(x)
            obj_y.extend(y)
            obj_z.extend(z)

            current_start = current_stop + timedelta(seconds=step_size_seconds)

    
    else:
        timerange = {'start':start_dt, 'stop':stop_dt, 'step':step}

        try:
            coord = get_horizons_coord(space_obj_code, timerange)
            coord = coord.transform_to(HeliocentricMeanEcliptic(obstime=coord.obstime))
            
            
            # remove leap seconds because datetime is not comaptible with them
            obj_time = [datetime.strptime(coord[i].obstime.strftime('%Y-%m-%d %H:%M')+':00', '%Y-%m-%d %H:%M:%S') for i in range(len(coord))]
            obj_r = coord.distance.value * u.au.to(u.km) 
            obj_lon = np.deg2rad(coord.lon.value)
            obj_lat = np.deg2rad(coord.lat.value)

            x,y,z = elevo_utils.convert_to_cartesian(  obj_lon,obj_lat,np.array(coord.distance.value* u.au.to(u.km)  ) )
            obj_x = x
            obj_y = y
            obj_z = z

        except ValueError as e:
            print(f"Error retrieving data for {space_obj} from {start_dt} to {stop_dt}: {e}")

            calc_times = [start_dt + timedelta(seconds=i*step_size_seconds) for i in range(int((stop_dt - start_dt).total_seconds()/step_size_seconds + 1))]
            calc_times = [datetime.strptime(calc_times[i].strftime('%Y-%m-%d %H:%M'+':00'), '%Y-%m-%d %H:%M:%S') for i in range(len(calc_times))]

            obj_time = calc_times
            obj_r = [np.nan]*int((stop_dt - start_dt).total_seconds()/step_size_seconds + 1)
            obj_lon = [np.nan]*int((stop_dt - start_dt).total_seconds()/step_size_seconds + 1)
            obj_lat = [np.nan]*int((stop_dt - start_dt).total_seconds()/step_size_seconds + 1)

            obj_x = [np.nan]*int((stop_dt - start_dt).total_seconds()/step_size_seconds + 1)
            obj_y = [np.nan]*int((stop_dt - start_dt).total_seconds()/step_size_seconds + 1)
            obj_z = [np.nan]*int((stop_dt - start_dt).total_seconds()/step_size_seconds + 1)

    
    coord_dict[space_obj_name] = {'time': np.array(obj_time), 'r': np.array(obj_r), 'lon': np.array(obj_lon), 'lat': np.array(obj_lat), 'x': np.array(obj_x), 'y': np.array(obj_y), 'z': np.array(obj_z)}
    with open(save_path+name, 'wb') as f:
        pickle.dump(coord_dict, f)


def load_positions_jpl(data_path, date_start,date_end,step, space_obj):

    file_path =  data_path+space_obj+"_"+step+"_"+date_start+"_"+date_end+'_HAE.pkl'
    with open(file_path, "rb") as f:
        pos = pickle.load(f)

    pos = pos[space_obj]
    
    time_array = pos['time']
    r_array    = pos['r']
    lon_array  = pos['lon']
    lat_array  = pos['lat']
    x_array = pos['x']
    y_array = pos['y']
    z_array = pos['z']


    return {'time': time_array.flatten(), 'r': r_array.flatten(), 'lon': lon_array.flatten(), 'lat': lat_array.flatten(), 'x': x_array.flatten(), 'y': y_array.flatten(), 'z': z_array.flatten()}


DATE_FMT = "%Y-%m-%d"  # adjust to match whatever format start_date/end_date actually use

def _parse_date(date_str):
    return datetime.strptime(date_str, DATE_FMT)

def _find_existing_files(save_path, name, resolution):
    """Locate cached files for this spacecraft/resolution and parse their date ranges."""
    pattern = re.compile(rf"^{re.escape(name)}_{re.escape(str(resolution))}_(.+)_(.+)_HAE\.pkl$")
    matches = []
    if not os.path.isdir(save_path):
        return matches
    for fname in os.listdir(save_path):
        m = pattern.match(fname)
        if m:
            f_start, f_end = m.group(1), m.group(2)
            try:
                matches.append((os.path.join(save_path, fname),
                                 _parse_date(f_start), _parse_date(f_end),
                                 f_start, f_end))
            except ValueError:
                continue  # filename didn't match expected date format, skip
    return matches

def _load_pkl(file_path, space_obj):
    with open(file_path, "rb") as f:
        pos = pickle.load(f)
    pos = pos[space_obj]
    return {k: np.asarray(v).flatten() for k, v in pos.items()}

def _save_pkl(file_path, space_obj, data_dict):
    with open(file_path, "wb") as f:
        pickle.dump({space_obj: data_dict}, f)

def _merge_data(dicts):
    """Concatenate multiple position dicts, sort by time, drop duplicate timestamps."""
    keys = dicts[0].keys()
    merged = {k: np.concatenate([d[k] for d in dicts]) for k in keys}
    order = np.argsort(merged['time'])
    for k in merged:
        merged[k] = merged[k][order]
    _, unique_idx = np.unique(merged['time'], return_index=True)
    unique_idx = np.sort(unique_idx)
    for k in merged:
        merged[k] = merged[k][unique_idx]
    return merged

def _slice_by_time(data_dict, start_dt, end_dt):
    times = data_dict['time']
    mask = (times >= np.datetime64(start_dt)) & (times <= np.datetime64(end_dt))
    return {k: v[mask] for k, v in data_dict.items()}

def ensure_positions_range(name, start_date, end_date, resolution, save_path='data/spc_pos/'):
    """
    Guarantees a cached file covering [start_date, end_date] for `name`, downloading
    only whatever sub-range is missing and merging with any existing overlapping
    file(s). Returns the position dict sliced to the exact requested range.
    """
    os.makedirs(save_path, exist_ok=True)
    req_start_dt, req_end_dt = _parse_date(start_date), _parse_date(end_date)

    existing = _find_existing_files(save_path, name, resolution)
    overlapping = [e for e in existing if e[1] <= req_end_dt and e[2] >= req_start_dt]

    # Case 1: an existing file already fully covers the requested range
    for file_path, f_start_dt, f_end_dt, *_ in overlapping:
        if f_start_dt <= req_start_dt and f_end_dt >= req_end_dt:
            return _slice_by_time(_load_pkl(file_path, name), req_start_dt, req_end_dt)

    # Case 2: nothing on disk overlaps at all -> plain download, as before
    if not overlapping:
        create_positions_file(name, start_date, end_date, step=resolution,
                               save_path=save_path, overwrite=False)
        return load_positions_jpl(save_path, start_date, end_date, resolution, name)

    # Case 3: partial coverage -> download only the missing gap(s), merge, rename
    overlapping.sort(key=lambda e: e[1])
    union_start_dt = min(req_start_dt, min(e[1] for e in overlapping))
    union_end_dt = max(req_end_dt, max(e[2] for e in overlapping))

    pieces = [_load_pkl(fp, name) for fp, *_ in overlapping]

    covered = sorted([(e[1], e[2]) for e in overlapping])
    gaps, cursor = [], union_start_dt
    for c_start, c_end in covered:
        if c_start > cursor:
            gaps.append((cursor, c_start))
        cursor = max(cursor, c_end)
    if cursor < union_end_dt:
        gaps.append((cursor, union_end_dt))

    for g_start_dt, g_end_dt in gaps:
        g_start, g_end = g_start_dt.strftime(DATE_FMT), g_end_dt.strftime(DATE_FMT)
        create_positions_file(name, g_start, g_end, step=resolution,
                               save_path=save_path, overwrite=False)
        pieces.append(load_positions_jpl(save_path, g_start, g_end, resolution, name))
        gap_file = os.path.join(save_path, f"{name}_{resolution}_{g_start}_{g_end}_HAE.pkl")
        if os.path.exists(gap_file):
            os.remove(gap_file)  # folded into the merged file below

    merged = _merge_data(pieces)

    for file_path, *_ in overlapping:
        os.remove(file_path)  # superseded by the merged, renamed file

    union_start, union_end = union_start_dt.strftime(DATE_FMT), union_end_dt.strftime(DATE_FMT)
    new_file_path = os.path.join(save_path, f"{name}_{resolution}_{union_start}_{union_end}_HAE.pkl")
    _save_pkl(new_file_path, name, merged)

    return _slice_by_time(merged, req_start_dt, req_end_dt)


def get_spcs_dates(start_date,end_date,names,resolution,timestemps):
    spcs = []
    for name in names:                 
        positions_dict = ensure_positions_range(name, start_date, end_date, resolution,
                                                 save_path='data/spc_pos/')
        spcs.append(SpaceObject(name, positions_dict['lon'], positions_dict['lat'], positions_dict['r'], 'HAE', start_date, timesteps=timestemps, time_resolution=resolution))
    return spcs
        

    