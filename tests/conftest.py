"""
Shared fixtures for the pyRES test suite.

Provides reusable test objects (PV panels, batteries, consumers, prosumers, RECs)
with known parameters so that expected values can be computed analytically.
"""

import pytest
import numpy as np
from src.rec_sim.PvPanels import PvPanels
from src.rec_sim.Bess import Bess
from src.rec_sim.Consumer import Consumer
from src.rec_sim.Prosumer import Prosumer
from src.rec_sim.Controller import Controller
from src.rec_sim.Rec import Rec
from src.rec_sim.Economics import Economics


# ---------------------------------------------------------------------------
# Constants used across tests
# ---------------------------------------------------------------------------
TIME_STEP_15MIN = 0.25       # 15 min in hours
TIME_STEP_1H = 1.0           # 1 hour
N_STEPS_DAY = 96             # 15-min steps in a day
N_STEPS_YEAR = 35040         # 15-min steps in a year


# ---------------------------------------------------------------------------
# PvPanels fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def pv_default():
    """PV panel with default NeON 2 parameters, 1x1 array (single module)."""
    return PvPanels(
        id='pv_test',
        cap_cost=1500,
        opex_cost=40,
        inc_year=0,
        inc_start_end=[0, 0],
        tax_year=0,
        n_series=1,
        n_parallel=1,
    )


@pytest.fixture
def pv_array_3x2():
    """PV array with 3 modules in series, 2 in parallel."""
    return PvPanels(
        id='pv_array',
        cap_cost=1500,
        opex_cost=40,
        inc_year=900,
        inc_start_end=[1, 10],
        tax_year=0,
        n_series=3,
        n_parallel=2,
        dc_ac_efficiency=0.97,
        mismatch_loss=0.02,
        wiring_loss=0.015,
        soiling_loss=0.03,
    )


@pytest.fixture
def pv_no_losses():
    """PV panel with all system losses disabled (for isolation testing)."""
    return PvPanels(
        id='pv_no_loss',
        cap_cost=1500,
        opex_cost=40,
        inc_year=0,
        inc_start_end=[0, 0],
        tax_year=0,
        n_series=1,
        n_parallel=1,
        dc_ac_efficiency=1.0,
        mismatch_loss=0.0,
        wiring_loss=0.0,
        soiling_loss=0.0,
    )


# ---------------------------------------------------------------------------
# Bess fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def bess_default():
    """Single battery module with default parameters."""
    return Bess(
        id='bess_test',
        cap=2.56,
        c_rate=1.0,
        soc_in=0.5,
        soc_max=0.8,
        soc_min=0.2,
        cap_cost=720,
        opex_cost=20,
        inc_year=0,
        inc_start_end=[0, 0],
        tax_year=0,
        eta_charge=0.95,
        eta_discharge=0.95,
        self_discharge_rate_per_hour=0.00004,
    )


@pytest.fixture
def bess_ideal():
    """Ideal battery with no losses (eta=1, no self-discharge)."""
    return Bess(
        id='bess_ideal',
        cap=10.0,
        c_rate=1.0,
        soc_in=0.5,
        soc_max=1.0,
        soc_min=0.0,
        cap_cost=500,
        opex_cost=10,
        inc_year=0,
        inc_start_end=[0, 0],
        tax_year=0,
        eta_charge=1.0,
        eta_discharge=1.0,
        self_discharge_rate_per_hour=0.0,
    )


@pytest.fixture
def bess_with_c_rate():
    """Battery with custom C-rate."""
    return Bess(
        id='bess_crate',
        cap=5.0,
        c_rate=0.5,
        soc_in=0.5,
        soc_max=0.9,
        soc_min=0.1,
        cap_cost=720,
        opex_cost=20,
        inc_year=0,
        inc_start_end=[0, 0],
        tax_year=0,
        eta_charge=0.95,
        eta_discharge=0.95,
    )


# ---------------------------------------------------------------------------
# Consumer fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def consumer_flat_1kw():
    """Consumer with flat 1 kW demand for 96 steps (1 day at 15-min)."""
    return Consumer(
        id='cons_flat',
        dem={'electricity': np.ones(N_STEPS_DAY) * 1.0}
    )


@pytest.fixture
def consumer_variable():
    """Consumer with a realistic daily profile (peak at noon)."""
    hours = np.arange(N_STEPS_DAY) * 0.25  # 0..23.75
    # Simple sinusoidal demand profile: 0.5-2.5 kW
    dem = 1.5 + np.sin(hours / 24 * 2 * np.pi - np.pi / 2)
    return Consumer(
        id='cons_var',
        dem={'electricity': dem}
    )


# ---------------------------------------------------------------------------
# Synthetic irradiance data (no PVGIS dependency)
# ---------------------------------------------------------------------------
@pytest.fixture
def synthetic_irradiance_day():
    """
    Synthetic irradiance data for 1 day (96 steps at 15-min).
    Bell-shaped beam, small diffuse, near-zero ground reflected.
    """
    steps = N_STEPS_DAY
    hours = np.arange(steps) * 0.25
    # Bell-shaped beam: peaks at noon (~800 W/m2), zero at night
    beam = np.maximum(0, 800 * np.sin((hours - 6) / 12 * np.pi))
    beam[hours < 6] = 0
    beam[hours > 18] = 0
    skydiff = beam * 0.15
    grounddiff = beam * 0.03
    t_amb = 20 + 5 * np.sin((hours - 6) / 24 * 2 * np.pi)
    wind = np.ones(steps) * 2.0
    return {
        'I_beam': beam,
        'I_skydiff': skydiff,
        'I_grounddiff': grounddiff,
        't_amb': t_amb,
        'wind_speed': wind,
        'n_steps': steps,
    }


# ---------------------------------------------------------------------------
# Pre-built prosumer and REC for integration tests
# ---------------------------------------------------------------------------
@pytest.fixture
def simple_prosumer(pv_no_losses, bess_ideal, consumer_flat_1kw, synthetic_irradiance_day):
    """
    A simple prosumer with 1 PV (no losses), 1 ideal BESS, 1 flat consumer.
    PV output is pre-computed from synthetic irradiance.
    """
    irr = synthetic_irradiance_day
    pv_no_losses.compute_output(
        slope=30,
        theta=None,
        I_beam=irr['I_beam'],
        I_skydiff=irr['I_skydiff'],
        I_grounddiff=irr['I_grounddiff'],
        t_amb=irr['t_amb'],
        wind_speed=irr['wind_speed'],
    )
    prosumer = Prosumer(
        id='pros_test',
        carriers=['electricity'],
        systems=[pv_no_losses],
        users=[consumer_flat_1kw],
        bess=[bess_ideal],
    )
    return prosumer


@pytest.fixture
def simple_rec(simple_prosumer, consumer_variable):
    """A simple REC with 1 prosumer and 1 additional consumer."""
    # Must run prosumer energy performance first
    simple_prosumer.energy_performance(time=TIME_STEP_15MIN)
    rec = Rec(
        id='rec_test',
        carriers=['electricity'],
        prosumers=[simple_prosumer],
        consumers=[consumer_variable],
    )
    return rec
