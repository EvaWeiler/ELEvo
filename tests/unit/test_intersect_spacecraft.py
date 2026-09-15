import datetime

import numpy as np
import pytest

from CME_class import CME
from elevo_utils import  calculate_arrival,get_boundary_indices,convert_to_cartesian,project_to_ellipse_axes,is_point_in_ellipsoid
from Space_object_class import SpaceObject


def make_cme(longitude=0.0, latitude=0.0, cme_a=None, cme_b=None, cme_c=None):
    cme = CME(
        half_width=np.deg2rad(45),
        longitude=longitude,
        latitude=latitude,
        tilt=0.0,
        f=0.7,
        initial_speed=850,
        initial_time=datetime.datetime(2025, 1, 1),
        initial_radius=21.5,
        feature_type='SH',
        source_of_info='test',
    )
    if cme_a is not None:
        cme.cme_a = cme_a
        cme.cme_b = cme_b
        cme.cme_c = cme_c
    return cme


class TestConvertToCartesian:
    def test_cardinal_points_in_ecliptic_plane(self):
        # tests if various longitudes map to the correct cartesian coordinates in the ecliptic plane (lat=0)
        r = 10.0
        lats = np.zeros(3)
        lons = np.deg2rad([0.0, 90.0, 180.0])
        x, y, z = convert_to_cartesian(lons, lats, r)
        np.testing.assert_allclose(x, [r, 0.0, -r], atol=1e-10)
        np.testing.assert_allclose(y, [0.0, r, 0.0], atol=1e-10)
        np.testing.assert_allclose(z, [0.0, 0.0, 0.0], atol=1e-10)

    def test_poles(self):
        # tests if the north and south poles map to the correct cartesian coordinates
        r = 5.0
        lons = np.deg2rad([0.0, 123.0])
        lats = np.deg2rad([90.0, -90.0])
        x, y, z = convert_to_cartesian(lons, lats, r)
        np.testing.assert_allclose(x, [0.0, 0.0], atol=1e-10)
        np.testing.assert_allclose(y, [0.0, 0.0], atol=1e-10)
        np.testing.assert_allclose(z, [r, -r], atol=1e-10)


class TestCmeOrthonormalBase:
    @pytest.mark.parametrize("lon_deg,lat_deg", [
        (0.0, 0.0), (45.0, 0.0), (30.0, 20.0), (200.0, -35.0), (10.0, 89.0),
    ])
    def test_orthonormal(self, lon_deg, lat_deg):
        # tests if base is actually orthonormal: each vector has unit length and is perpendicular to the others
        cme = make_cme(longitude=np.deg2rad(lon_deg), latitude=np.deg2rad(lat_deg))
        vectors = cme.cme_orthonormal_base()
        for v in vectors:
            np.testing.assert_allclose(np.linalg.norm(v), 1.0, atol=1e-10)
        for i in range(3):
            for j in range(i + 1, 3):
                assert abs(np.dot(vectors[i], vectors[j])) < 1e-10

class TestProjectToEllipseAxes:

    def test_shape_and_broadcast_multiple_timesteps_and_ensemble(self):
        # tests if the output shape and dot product with the base vectors is correct for multiple timesteps and ensemble members
        cme = make_cme(longitude=np.deg2rad(30))
        normal_base = cme.cme_orthonormal_base()
        n_t, n_ens = 2, 3
        rng = np.random.default_rng(0)
        center = rng.normal(size=(3, n_t, n_ens))
        pos = rng.normal(size=(3, n_t))

        comps = project_to_ellipse_axes(pos, center, normal_base)
        assert len(comps) == 3
        for c in comps:
            assert c.shape == (n_t, n_ens)

        for t in range(n_t):
            for e in range(n_ens):
                rel = pos[:, t] - center[:, t, e]
                for axis_idx, comp in enumerate(comps):
                    assert comp[t, e] == pytest.approx(np.dot(rel, normal_base[axis_idx]))


class TestIsPointInEllipsoid:
    def test_center_is_inside(self):
        a = np.array([[2.0]])
        b = np.array([[1.0]])
        projected = (np.array([[0.0]]), np.array([[0.0]]), np.array([[0.0]]))
        assert is_point_in_ellipsoid(projected, a, b).all()

class TestCalculateIntersection:
    def test_raises_without_ellipse_parameters(self):
        cme = make_cme()
        space_object = SpaceObject(
            'sc', np.array([0.0]), np.array([0.0]), np.array([1.0]),
            'HAE', cme.initial_time, np.array([0.0]), 10.0,
        )
        with pytest.raises(ValueError):
            space_object.calculate_intersection(cme)

class TestGetBoundaryIndices:
    def test_never_always_and_normal_entry_columns(self):
        # col0: never intersects, col1: inside from t=0, col2: normal
        intersection = np.array([
            [False, True, False],
            [False, True, False],
            [False, False, True],
            [False, False, True],
            [False, False, False],
        ])
        before, first, last, after = get_boundary_indices(intersection)

        assert before[0] is None and first[0] is None and last[0] is None and after[0] is None

        assert first[1] == 0
        assert before[1] == 0
        assert last[1] == 1
        assert after[1] == 2

        assert before[2] == 1
        assert first[2] == 2
        assert last[2] == 3
        assert after[2] == 4

    def test_single_timestep_tangent_graze(self):
        intersection = np.array([[False], [False], [True], [False], [False]])
        before, first, last, after = get_boundary_indices(intersection)
        assert first[0] == 2
        assert last[0] == 2
        assert before[0] == 1
        assert after[0] == 3
