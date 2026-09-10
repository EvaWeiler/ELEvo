import datetime
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from ELEvo import CME, calculate_ellipse_parameters, create_ensemble, propagate_cme


@dataclass
class SpaceObject:
    """Represent a celestial object.

    Attributes
    ----------
    name : str
        Object name.
    longitude: array-like
        Longitude in the specified coordinate system, in radians.
    latitude : array-like
        Latitude in the specified coordinate system, in radians.
    radial_distance : array-like
        Radial distance from the Sun, in km.
    coordinate_system : str
        Name of the coordinate system used for the position values.
    initial_time : datetime.datetime
        Initial time corresponding to the first timestep of the position values.
    timesteps : array-like
        Time grid associated with the position values, in seconds since initial time.
    time_resolution : float
        Resolution of time grid, in minutes.
    """

    name: str
    longitude: np.ndarray
    latitude: np.ndarray
    radial_distance: np.ndarray
    coordinate_system: str
    initial_time: datetime.datetime
    timesteps: np.ndarray
    time_resolution: float

def convert_to_cartesian(longitude, latitude, radial_distance):
    """Convert spherical coordinates to Cartesian coordinates.

    Parameters
    ----------
    longitude : array-like
        Longitude in radians.
    latitude : array-like
        Latitude in radians.
    radial_distance : array-like
        Distance from the Sun.
    
    Returns
    -------
    x, y, z : array-like
        Cartesian coordinates corresponding to the input spherical coordinates.
    """

    x = radial_distance*np.cos(latitude)*np.cos(longitude)
    y = radial_distance*np.cos(latitude)*np.sin(longitude)
    z = radial_distance*np.sin(latitude)

    return x, y, z

def project_to_ellipse_axes(space_obj_pos, ellipse_center, normal_base):
    """Project a point onto the axes of an ellipse defined by its center and normal base.

    Parameters
    ----------
    point : array-like
        Cartesian coordinates of the point to project. Dimension is (3, n_timesteps).
    ellipse_center : array-like
        Cartesian coordinates of the ellipse center. Dimension is (3, n_timesteps, n_ensemble_members).
    normal_base : array-like
        Orthonormal base vectors defining the orientation of the ellipse. Dimension is (3, 3).

    Returns
    -------
    projected_coords : array-like
        Coordinates of the point projected onto the ellipse axes. Dimension is (3,).
    """

    relative_position = np.array(space_obj_pos)[:,:,None] - np.array(ellipse_center)

    # einsum is used to compute the dot product for each timestep and ensemble member (i-dimension = 3, j-dimension = n_timesteps, k-dimension = n_ensemble_members)
    component_along_x = np.einsum('ijk,i->jk', relative_position, normal_base[0])
    component_along_y = np.einsum('ijk,i->jk', relative_position, normal_base[1])
    component_along_z = np.einsum('ijk,i->jk', relative_position, normal_base[2])

    
    return component_along_x, component_along_y, component_along_z

def is_point_in_ellipsoid(projected_points, a, b):
    """Check if a point is inside an ellipse. This assumes that the point has already been projected onto the axes of the ellipse.
    Only works for ellipses where b=c.

    Parameters
    ----------
    projected_points : array-like
        Coordinates of the point projected onto the ellipse axes. Dimension is (3, n_timesteps, n_ensemble_members).
    a : array-like
        Semi-major axis of the ellipse. Dimension is (n_timesteps, n_ensemble_members).
    b : array-like
        Semi-minor axis of the ellipse. Dimension is (n_timesteps, n_ensemble_members).
    Returns
    -------
    array-like
        Boolean array that is True where the point is inside the ellipse, False otherwise.
    """

    component_along_x, component_along_y, component_along_z = projected_points

    return (component_along_x/a)**2 + (component_along_y/b)**2 + (component_along_z/b)**2 <= 1

def cme_orthonormal_base(cme):
    """Compute orthonormal base for CME coordinates.

    Parameters
    ----------
    cme : CME
        The CME object containing the longitude.

    Returns
    -------
    array-like
        Set of vectors spanning an orthonormal base for the CME. Dimension is (3,3).
    """

    # Vector in the ecliptic plane, perpendicular to the CME's longitude direction
    return np.array([-np.sin(cme.longitude), np.cos(cme.longitude), 0.0]), np.array([np.cos(cme.longitude), np.sin(cme.longitude), 0.0]), np.array([0.0, 0.0, 1.0])

def calculate_intersection(cme, space_object):
    """Determine if a spacecraft is inside the CME ellipse at each timestep.
    Depends on calculate_ellipse_parameters being called first to compute the ellipse parameters.

    Parameters
    ----------
    cme : CME
        The CME object containing the ellipse parameters.
    space_object : SpaceObject
        The spacecraft object containing its position.

    Returns
    -------
    intersection_result: array-like
        Boolean array that is True where the spacecraft is inside the CME ellipse, False otherwise.
    """

    if not hasattr(cme, 'cme_a') or not hasattr(cme, 'cme_b') or not hasattr(cme, 'cme_c'):
        raise ValueError("CME ellipse parameters not calculated. Call calculate_ellipse_parameters(cme) first.")

    normal_base_vectors = cme_orthonormal_base(cme)

    sc_cartesian = convert_to_cartesian(space_object.longitude, space_object.latitude, space_object.radial_distance)
    cme_cartesian = convert_to_cartesian(cme.longitude, cme.latitude, cme.cme_c)

    projected_sc_components = project_to_ellipse_axes(sc_cartesian, cme_cartesian, normal_base_vectors)
    intersection_result = is_point_in_ellipsoid(projected_sc_components, cme.cme_a, cme.cme_b)

    return intersection_result

def calculate_time_intersection(intersection_result, timesteps, initial_time):
    """For each ensemble member, return the first and last times where the
    spacecraft is inside the CME, or (None, None) if it never is.

    Parameters
    ----------
    intersection_result : array-like
        Boolean array that is True where the spacecraft is inside the CME ellipse, False otherwise. Has shape (n_timesteps, n_ensemble_members).
    timesteps : array-like
        Time grid associated with the position values, given in seconds since initial_time. Has shape (n_timesteps, n_ensemble_members).
    initial_time : datetime.datetime
        The initial time of the CME corresponding to the first timestep.

    Returns
    -------
    time_bounds : array-like
        Array of shape (n_ensemble_members, 2) where each row contains the first and last times (as datetime objects) where the spacecraft is inside the CME, or (None, None) if it never is.
    """

    n_ensemble = intersection_result.shape[1]
    ensemble_idx = np.arange(n_ensemble)
    has_any = intersection_result.any(axis=0)

    first_idx = intersection_result.argmax(axis=0)
    last_idx = intersection_result.shape[0] - 1 - intersection_result[::-1].argmax(axis=0)

    first_seconds = timesteps[first_idx, ensemble_idx][has_any]
    last_seconds = timesteps[last_idx, ensemble_idx][has_any]

    base_time = np.datetime64(initial_time)
    first_offsets = np.round(first_seconds * 1e6).astype('int64').astype('timedelta64[us]')
    last_offsets = np.round(last_seconds * 1e6).astype('int64').astype('timedelta64[us]')

    time_bounds = np.full((n_ensemble, 2), None, dtype=object)
    time_bounds[has_any, 0] = (base_time + first_offsets).astype('datetime64[us]').astype(object)
    time_bounds[has_any, 1] = (base_time + last_offsets).astype('datetime64[us]').astype(object)

    return time_bounds

def _ellipsoid_surface_points(center, a, b, normal_base, n_theta=25, n_phi=25):
    """Generate points on the surface of an ellipsoid defined by its center, semi-major axis a, semi-minor axis b, and normal vector.

    Parameters
    ----------
    center : array-like
        Cartesian coordinates of the ellipsoid center. Dimension is (3,).
    a : float
        Semi-major axis of the ellipsoid.
    b : float
        Semi-minor axis of the ellipsoid.
    normal_base : array-like
        Orthonormal base for the CME. Dimension is (3,3).
    n_theta : int
        Number of points along the polar angle (theta).
    n_phi : int
        Number of points along the azimuthal angle (phi).

    Returns
    -------
    coords : list of array-like
        List containing the x, y, z coordinates of the ellipsoid surface points. Each array has shape (n_phi, n_theta).
    """
    u, v , w = normal_base

    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi)
    theta, phi = np.meshgrid(theta, phi)

    axial = a * np.cos(theta)
    radial = b * np.sin(theta)

    coords = []
    for i in range(3):
        coords.append(
            center[i]
            + axial * u[i]
            + radial * (np.cos(phi) * w[i] + np.sin(phi) * v[i])
        )
    return coords


def plot_intersection_debug(cme, space_object, intersection_result, ensemble_idx=0, time_indices=None, n_snapshots=5):
    """Render the CME spheroid and the spacecraft position at a handful of
    timesteps, for a single ensemble member, colouring the spacecraft green when
    intersection_result says it is inside and red otherwise.

    Parameters
    ----------
    cme : CME
        The CME object containing the ellipse parameters.
    space_object : SpaceObject
        The spacecraft object containing its position.
    intersection_result : array-like
        Boolean array that is True where the spacecraft is inside the CME ellipse, False otherwise. Has shape (n_timesteps, n_ensemble_members).
    ensemble_idx : int
        Index of the ensemble member to visualize.
    time_indices : array-like, optional
        Indices of the timesteps to visualize. If None, n_snapshots evenly spaced timesteps will be chosen.
    n_snapshots : int
        Number of timesteps to visualize if time_indices is None.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib figure object containing the plots.
    """

    n_time = cme.cme_r_ensemble.shape[0]
    if time_indices is None:
        time_indices = np.linspace(0, n_time - 1, n_snapshots, dtype=int)

    normal_base = cme_orthonormal_base(cme)

    sc_x, sc_y, sc_z = convert_to_cartesian(space_object.longitude, space_object.latitude, space_object.radial_distance)

    fig = plt.figure(figsize=(4.5 * len(time_indices), 4.5))

    for i, t in enumerate(time_indices):
        ax = fig.add_subplot(1, len(time_indices), i + 1, projection='3d')

        center = convert_to_cartesian(cme.longitude, cme.latitude, cme.cme_c[t, ensemble_idx])
        a = cme.cme_a[t, ensemble_idx]
        b = cme.cme_b[t, ensemble_idx]

        x, y, z = _ellipsoid_surface_points(center, a, b, normal_base)
        ax.plot_surface(x, y, z, color='tab:orange', alpha=0.25, linewidth=0)

        ax.plot([0, center[0]], [0, center[1]], [0, center[2]], 'k--', linewidth=0.8)
        ax.scatter(0, 0, 0, color='gold', s=100, label='Sun')
        ax.scatter(*center, color='tab:orange', s=30, label='CME apex')

        is_inside = bool(intersection_result[t, ensemble_idx])
        sc_color = 'tab:green' if is_inside else 'tab:red'
        ax.scatter(sc_x[t], sc_y[t], sc_z[t], color=sc_color, s=60, label=space_object.name)

        ax.set_title(f"t index {t}\n{'INSIDE' if is_inside else 'outside'}")
        ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')

        span = np.array([center, [sc_x[t], sc_y[t], sc_z[t]]])
        max_range = max(np.abs(span).max(), a, b) * 1.2
        ax.set_xlim(-max_range, max_range)
        ax.set_ylim(-max_range, max_range)
        ax.set_zlim(-max_range, max_range)

        if i == 0:
            ax.legend(loc='upper left', fontsize=8)

    fig.tight_layout()
    return fig

if __name__ == "__main__":

    initial_time = datetime.datetime(2025,10,12,12,58)

    cme = CME(half_width=np.deg2rad(45),
              longitude=np.deg2rad(0),
              latitude=np.deg2rad(0),
              tilt=0,
              f=0.7,
              initial_speed=850,
              initial_time=initial_time,
              initial_radius=21.5,
              feature_type='SE',
              source_of_info='Dummy values')

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
    gamma_array, ambient_wind_array, cme = create_ensemble(400,1.0e-7,cme,std_ensemble,nb_ensemble=20000,method_type='skewed')
    cme = propagate_cme(gamma_array, ambient_wind_array,cme,days_duration=10,minute_resolution=10)
    cme = calculate_ellipse_parameters(cme)

    earth_r = np.zeros(np.shape(cme.cme_r_ensemble)[0])
    earth_r[:] = 150*10**6
    # earth_lon = np.zeros(np.shape(cme.cme_r_ensemble)[0])
    earth_lon = np.linspace(0,45.0,np.shape(cme.cme_r_ensemble)[0])
    earth_lon = np.deg2rad(earth_lon)
    earth_lat = np.zeros(np.shape(cme.cme_r_ensemble)[0])
    earth_timegrid = cme.ensemble_timesteps
    earth_time_resolution = cme.ensemble_time_resolution

    earth_object = SpaceObject('Earth', earth_lon, earth_lat, earth_r, 'HAE', cme.initial_time, timesteps=earth_timegrid, time_resolution=earth_time_resolution)

    intersection = calculate_intersection(cme, earth_object)
    time_bounds = calculate_time_intersection(intersection, cme.ensemble_timesteps, cme.initial_time)
    debug_ensemble_idx = -1
    # compute how many ensemble members evetually arrive at earth
    n_arrival = np.sum(time_bounds[:,0] != None)
    print(f"Number of ensemble members that arrive at Earth: {n_arrival} / {time_bounds.shape[0]} ({n_arrival/time_bounds.shape[0]*100:.2f}%)")
    time_bounds = time_bounds[debug_ensemble_idx]
    print(f"Ensemble member {debug_ensemble_idx} first inside index: {time_bounds[0]}, last inside index: {time_bounds[1]}")
    plot_intersection_debug(cme, earth_object, intersection, ensemble_idx=debug_ensemble_idx)
    plt.show()