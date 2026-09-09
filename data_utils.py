import os 
import datetime
import requests
import json 
import ELEvo
import numpy as np 
import astropy.units as u
from astropy.coordinates import SkyCoord, HeliocentricMeanEcliptic
from astropy.time import Time
import sunpy.coordinates  
from astropy import constants as const

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


def convert_HEEQ_to_HEA(long,lat,radius,time_obs):
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

def load_donki_cmes(path_json):
    # Opening JSON file
    with open(path_json) as json_file:
        donki_data = json.load(json_file)

    CMEs = []
    for index, row in enumerate(donki_data):
        for analysis in row['cmeAnalyses']:
            if analysis['longitude'] is not None and analysis['latitude'] is not None:
                long,lat,radius = convert_HEEQ_to_HEA(analysis['longitude'],analysis['latitude'],21.5,analysis['time21_5'])
                cme = ELEvo.CME(analysis['halfAngle'],
                                long,
                                lat,
                                0.0,
                                0.7,
                                analysis['speed'],
                                analysis['time21_5'],
                                21.5,
                                analysis['featureCode'],
                                'DONKI'
                                )
                CMEs.append(cme)
    return CMEs

def download_donki_cmes(date_start,date_end,data_type='CME',file_path='data/donki_data.json'):
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
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(response.text)

    except requests.RequestException as e:
        raise RuntimeError(f"Failed to download DONKI data: {e}")


if __name__ == "__main__":
    download_donki_cmes(datetime.datetime(2026,1,12),datetime.datetime(2026,1,17))
    print(len(load_donki_cmes('data/donki_data.json')))