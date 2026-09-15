import datetime

import numpy as np
import pytest

from CME_class import CME
from elevo_utils import  calculate_arrival,get_boundary_indices
from space_object_class import SpaceObject


def build_cme(half_width=45., longitude=0.0, latitude=0.0, tilt=0.0, f=0.7, initial_speed=850, initial_time=datetime.datetime(2025, 10, 12, 12, 58), initial_radius=21.5):
    return CME(
        half_width=np.deg2rad(half_width),
        longitude=np.deg2rad(longitude),
        latitude=np.deg2rad(latitude),
        tilt=tilt,
        f=f,
        initial_speed=initial_speed,
        initial_time=initial_time,
        initial_radius=initial_radius,
        feature_type='SH',
        source_of_info='test',
    )


DEFAULT_STD_ENSEMBLE = {
    'sw': 50, 'gamma': 0.025, 'half_width': 0, 'longitude': 0.0,
    'latitude': 0.0, 'tilt': 0, 'initial_speed': 50, 'initial_radius': 1,
}
ZERO_STD_ENSEMBLE = dict.fromkeys(DEFAULT_STD_ENSEMBLE, 0)


def run_pipeline(cme, spacecraft_longitude_deg, spacecraft_latitude_deg, nb_ensemble=20, days_duration=14, std_ensemble=DEFAULT_STD_ENSEMBLE):

    cme.initialize_ensemble(
        400, 0.2, std_ensemble, nb_ensemble=nb_ensemble, random_seed=42, method_type='normal',
    )
    cme.propagate_cme()
    cme.calculate_ellipse_parameters()

    n_time = cme.cme_r_ensemble.shape[0]
    sc_lon = np.full(n_time, np.deg2rad(spacecraft_longitude_deg))
    sc_lat =np.full(n_time, np.deg2rad(spacecraft_latitude_deg))
    sc_r = np.full(n_time, 150e6)  # ~1 AU in km

    space_object = SpaceObject(
        'sc', sc_lon, sc_lat, sc_r, 'HAE', cme.initial_time,
        timesteps=cme.ensemble_timesteps, time_resolution=cme.ensemble_time_resolution,
    )

    intersection = space_object.calculate_intersection(cme)

    bound_idxs = get_boundary_indices(intersection)
    arrival_times, arrival_speeds = calculate_arrival(
        bound_idxs, cme.ensemble_timesteps, cme.cme_v_ensemble, cme.initial_time,
    )
    return cme, intersection, bound_idxs, arrival_times, arrival_speeds


ARRIVAL_TIME_RANGE = (1.5, 5.0)  # days from initial_time
SPEED_RANGE = (400., 700.)  # km/s, decelerating from 850 km/s initial speed

ARRIVAL_CASES = [
    # half_width=45, longitude=0, latitude=0
    pytest.param(build_cme(half_width=45., longitude=0.0, latitude=0.0), 0., 0., True, ARRIVAL_TIME_RANGE, id="hw45-lon0-lat0_hit-center"),
    pytest.param(build_cme(half_width=45., longitude=0.0, latitude=0.0), 20., 0., True, ARRIVAL_TIME_RANGE, id="hw45-lon0-lat0_hit-lon-offset20"),
    pytest.param(build_cme(half_width=45., longitude=0.0, latitude=0.0), 0., 20., True, ARRIVAL_TIME_RANGE, id="hw45-lon0-lat0_hit-lat-offset20"),
    pytest.param(build_cme(half_width=45., longitude=0.0, latitude=0.0), 90., 0., False, None, id="hw45-lon0-lat0_miss-lon90"),
    pytest.param(build_cme(half_width=45., longitude=0.0, latitude=0.0), 0., 60., False, None, id="hw45-lon0-lat0_miss-lat60"),
    pytest.param(build_cme(half_width=45., longitude=0.0, latitude=0.0), 180., 0., False, None, id="hw45-lon0-lat0_miss-opposite"),
    # half_width=30, longitude=0, latitude=0
    pytest.param(build_cme(half_width=30., longitude=0.0, latitude=0.0), 0., 0., True, ARRIVAL_TIME_RANGE, id="hw30-lon0-lat0_hit-center"),
    pytest.param(build_cme(half_width=30., longitude=0.0, latitude=0.0), 45., 0., False, None, id="hw30-lon0-lat0_miss-lon45-inside-hw45-but-not-hw30"),
    # half_width=45, longitude=30, latitude=0
    pytest.param(build_cme(half_width=45., longitude=30.0, latitude=0.0), 30., 0., True, ARRIVAL_TIME_RANGE, id="hw45-lon30-lat0_hit-center"),
    pytest.param(build_cme(half_width=45., longitude=30.0, latitude=0.0), 90., 0., False, None, id="hw45-lon30-lat0_miss-offset60"),
    # half_width=45, longitude=200, latitude=0
    pytest.param(build_cme(half_width=45., longitude=200.0, latitude=0.0), 200., 0., True, ARRIVAL_TIME_RANGE, id="hw45-lon200-lat0_hit-center"),
    pytest.param(build_cme(half_width=45., longitude=200.0, latitude=0.0), 20., 0., False, None, id="hw45-lon200-lat0_miss-opposite"),
    # half_width=45, longitude=0, latitude=30
    pytest.param(build_cme(half_width=45., longitude=0.0, latitude=30.0), 0., 30., True, ARRIVAL_TIME_RANGE, id="hw45-lat30_hit-center"),
    pytest.param(build_cme(half_width=45., longitude=0.0, latitude=30.0), 0., -30., False, None, id="hw45-lat30_miss-opposite-lat"),
    # half_width=45, longitude=0, latitude=-30
    pytest.param(build_cme(half_width=45., longitude=0.0, latitude=-30.0), 0., -30., True, ARRIVAL_TIME_RANGE, id="hw45-latneg30_hit-center"),
    pytest.param(build_cme(half_width=45., longitude=0.0, latitude=-30.0), 0., 30., False, None, id="hw45-latneg30_miss-opposite-lat"),
]


class TestArrivalApproximate:
    @pytest.mark.parametrize(
        "test_cme, spacecraft_longitude_deg, spacecraft_latitude_deg, expect_arrival, time_range_days",
        ARRIVAL_CASES,
    )
    def test_spacecraft_arrivals(self, test_cme, spacecraft_longitude_deg, spacecraft_latitude_deg, expect_arrival, time_range_days):
        cme, intersection, bound_idxs, arrival_times, arrival_speeds = run_pipeline(test_cme, spacecraft_longitude_deg, spacecraft_latitude_deg)

        n_time, n_ens = intersection.shape
        assert bound_idxs.shape == (4, n_ens)
        assert arrival_times.shape == (n_ens,)
        assert arrival_speeds.shape == (n_ens,)

        has_arrival = bound_idxs[0] != None

        if not expect_arrival:
            assert not intersection.any()
            assert not has_arrival.any()
            assert all(v is None for v in bound_idxs.flatten())
            assert all(v is None for v in arrival_times)
            assert all(v is None for v in arrival_speeds)
            return

        assert has_arrival.all()

        min_days, max_days = time_range_days
        for i in range(n_ens):
            before, first, last, after = (int(bound_idxs[k, i]) for k in range(4))
            assert before <= first <= last <= after
            assert arrival_times[i] is not None
            arrival_day = (arrival_times[i] - cme.initial_time).total_seconds() / 86400.
            assert min_days <= arrival_day <= max_days
            assert arrival_speeds[i] is not None
            assert SPEED_RANGE[0] <= arrival_speeds[i] <= SPEED_RANGE[1]


# Frozen reference values, captured from ELEvo at commit 1676083 with a single ensemble member. A failure here means model output changed.
# The centre-aligned cases share identical values because the spacecraft sits on the CME propagation axis.

REFERENCE_CASES = [
    pytest.param(build_cme(half_width=45., longitude=0.0, latitude=0.0), 0., 0., datetime.datetime(2025, 10, 14, 22, 13), 557.6238756150044, id="hw45-lon0-lat0_hit-center"),
    pytest.param(build_cme(half_width=45., longitude=0.0, latitude=0.0), 20., 0., datetime.datetime(2025, 10, 15, 0, 13), 554.1255556480455, id="hw45-lon0-lat0_hit-lon-offset20"),
    pytest.param(build_cme(half_width=45., longitude=0.0, latitude=0.0), 0., 20., datetime.datetime(2025, 10, 15, 7, 43), 542.2835956501187, id="hw45-lon0-lat0_hit-lat-offset20"),
    pytest.param(build_cme(half_width=30., longitude=0.0, latitude=0.0), 0., 0., datetime.datetime(2025, 10, 14, 22, 13), 557.6238756150044, id="hw30-lon0-lat0_hit-center"),
    pytest.param(build_cme(half_width=45., longitude=30.0, latitude=0.0), 30., 0., datetime.datetime(2025, 10, 14, 22, 13), 557.6238756150044, id="hw45-lon30-lat0_hit-center"),
    pytest.param(build_cme(half_width=45., longitude=200.0, latitude=0.0), 200., 0., datetime.datetime(2025, 10, 14, 22, 13), 557.6238756150044, id="hw45-lon200-lat0_hit-center"),
    pytest.param(build_cme(half_width=45., longitude=0.0, latitude=30.0), 0., 30., datetime.datetime(2025, 10, 14, 22, 13), 557.6238756150044, id="hw45-lat30_hit-center"),
    pytest.param(build_cme(half_width=45., longitude=0.0, latitude=-30.0), 0., -30., datetime.datetime(2025, 10, 14, 22, 13), 557.6238756150044, id="hw45-latneg30_hit-center"),
]


class TestArrivalReference:
    @pytest.mark.parametrize(
        "test_cme, spacecraft_longitude_deg, spacecraft_latitude_deg, expected_arrival_time, expected_arrival_speed",
        REFERENCE_CASES,
    )
    def test_single_member_arrival_matches_reference(self, test_cme, spacecraft_longitude_deg, spacecraft_latitude_deg, expected_arrival_time, expected_arrival_speed):
        _, _, _, arrival_times, arrival_speeds = run_pipeline(
            test_cme, spacecraft_longitude_deg, spacecraft_latitude_deg,
            nb_ensemble=1, std_ensemble=ZERO_STD_ENSEMBLE,
        )

        assert arrival_times.shape == (1,)
        assert arrival_speeds.shape == (1,)
        assert arrival_times[0] == expected_arrival_time
        assert arrival_speeds[0] == pytest.approx(expected_arrival_speed, rel=1e-9)
