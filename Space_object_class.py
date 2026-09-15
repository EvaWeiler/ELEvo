import datetime
import numpy as np 
from dataclasses import dataclass
import elevo_utils

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

    def calculate_intersection(self,cme):
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

        normal_base_vectors = cme.cme_orthonormal_base()

        sc_cartesian = elevo_utils.convert_to_cartesian(self.longitude, self.latitude, self.radial_distance)
        cme_cartesian = elevo_utils.convert_to_cartesian(cme.longitude, cme.latitude, cme.cme_c)

        projected_sc_components = elevo_utils.project_to_ellipse_axes(sc_cartesian, cme_cartesian, normal_base_vectors)
        intersection_result = elevo_utils.is_point_in_ellipsoid(projected_sc_components, cme.cme_a, cme.cme_b)

        return intersection_result

    