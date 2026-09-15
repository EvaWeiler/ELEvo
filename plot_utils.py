import matplotlib.pyplot as plt 
import numpy as np 
import elevo_utils

def plot_spcs_cmepropagation(spcs,cme):


    fig, ax = plt.subplots(1, 1, subplot_kw={'projection': 'polar'},layout='constrained')
    for spc in spcs:
        ax.plot(spc.longitude,spc.radial_distance,label=spc.name)
    ax.plot(cme.longitude *np.ones(cme.cme_r_ensemble.shape).flatten(),cme.cme_r_ensemble.flatten(),label="cme")
    plt.legend()
    plt.show()

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

    u, v ,w = normal_base

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

    normal_base = cme.cme_orthonormal_base()

    sc_x, sc_y, sc_z = elevo_utils.convert_to_cartesian(space_object.longitude, space_object.latitude, space_object.radial_distance)

    fig = plt.figure(figsize=(4.5 * len(time_indices), 4.5))

    for i, t in enumerate(time_indices):
        ax = fig.add_subplot(1, len(time_indices), i + 1, projection='3d')

        center = elevo_utils.convert_to_cartesian(cme.longitude, cme.latitude, cme.cme_c[t, ensemble_idx])
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
    plt.show()