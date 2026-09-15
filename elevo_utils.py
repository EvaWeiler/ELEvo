import numpy as np
from scipy.stats import skewnorm


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


def create_time_grid(nb_ensembles,days_duration,minute_resolution):
    n_timesteps = int(days_duration * 24 * 60 / minute_resolution)
    timesteps = np.arange(n_timesteps+1) * minute_resolution * 60.0
    timesteps = np.repeat(timesteps[None,:],repeats=nb_ensembles,axis=0)
    timesteps = np.transpose(timesteps)
    return timesteps


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

def get_boundary_indices(intersection_result):
    """For each ensemble member, find the timestep indices bracketing the first and last
    intersection with the CME, i.e. the timestep just before the first crossing, the first crossing,
    the last crossing, and the timestep just after the last crossing. Ensemble members that never
    intersect the CME get (None, None, None, None).

    Parameters
    ----------
    intersection_result : array-like
        Boolean array that is True where the spacecraft is inside the CME ellipse, False otherwise. Has shape (n_timesteps, n_ensemble_members).

    Returns
    -------
    array-like
        Object array of shape (4, n_ensemble_members): [before_first, first, last, after_last] timestep indices, or None in all four rows for ensemble members that never intersect the CME.
    """

    n_timesteps = intersection_result.shape[0]
    has_any = intersection_result.any(axis=0)

    first_idx = intersection_result.argmax(axis=0)
    last_idx = n_timesteps - 1 - intersection_result[::-1].argmax(axis=0)

    before_first_idx = np.maximum(first_idx - 1, 0)
    after_last_idx = np.minimum(last_idx + 1, n_timesteps - 1)

    boundary_idx = np.stack([before_first_idx, first_idx, last_idx, after_last_idx]).astype(object)
    boundary_idx[:, ~has_any] = None

    return boundary_idx

def calculate_arrival(bound_idx, timesteps, speeds, initial_time):
    """For each ensemble member, compute the arrival time and arrival speed at the CME boundary:
    the average of the timestep just before the first crossing into the CME and the first
    crossing itself. Returns None for ensemble members that never intersect the CME.

    Parameters
    ----------
    bound_idx : array-like
        Boundary timestep indices as returned by get_boundary_indices, shape (4, n_ensemble_members).
        Only the first two rows (before_first, first) are used.
    timesteps : array-like
        Time grid associated with the position values, given in seconds since initial_time. Has shape (n_timesteps, n_ensemble_members).
    speeds : array-like
        CME speed at each timestep and ensemble member. Has shape (n_timesteps, n_ensemble_members).
    initial_time : datetime.datetime
        The initial time of the CME corresponding to the first timestep.

    Returns
    -------
    arrival_times : array-like
        Object array of shape (n_ensemble_members,): arrival time as a datetime object, or None.
    arrival_speeds : array-like
        Object array of shape (n_ensemble_members,): arrival speed, or None.
    """

    arrival_seconds = average_boundary_values(bound_idx[:2], timesteps)
    arrival_speeds = average_boundary_values(bound_idx[:2], speeds)

    has_any = arrival_seconds != None
    base_time = np.datetime64(initial_time)
    arrival_offsets = offsets_from_seconds(arrival_seconds[has_any].astype(float))

    arrival_times = np.full(len(arrival_seconds), None, dtype=object)
    arrival_times[has_any] = (base_time + arrival_offsets).astype('datetime64[us]').astype(object)

    return arrival_times, arrival_speeds


def offsets_from_seconds(seconds):
    """Convert an array of seconds into microsecond-precision numpy timedeltas.

    Parameters
    ----------
    seconds : array-like
        Seconds to convert.

    Returns
    -------
    array-like
        Array of dtype timedelta64[us], same shape as the input.
    """

    return np.round(seconds * 1e6).astype('int64').astype('timedelta64[us]')

def average_boundary_values(bound_idx_pair, values):
    """Average a per-timestep array at two boundary timestep indices, for each ensemble member.

    Parameters
    ----------
    bound_idx_pair : array-like
        Two rows of timestep indices, shape (2, n_ensemble_members), as returned by (a slice of)
        get_boundary_indices. None in both rows means no intersection for that ensemble member.
    values : array-like
        Per-timestep data to average, e.g. seconds-since-initial_time or CME speed. Shape
        (n_timesteps, n_ensemble_members).

    Returns
    -------
    array-like
        Object array of shape (n_ensemble_members,): the average of values at the two boundary
        indices, or None for ensemble members with no intersection.
    """

    n_ensemble = bound_idx_pair.shape[1]
    ensemble_idx = np.arange(n_ensemble)
    has_any = bound_idx_pair[0] != None

    valid_idx = bound_idx_pair[:, has_any].astype(int)
    boundary_values = values[valid_idx, ensemble_idx[has_any]]

    averages = np.full(n_ensemble, None, dtype=object)
    averages[has_any] = boundary_values.mean(axis=0)

    return averages