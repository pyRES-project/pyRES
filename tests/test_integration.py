"""
Integration tests for the full pyRES simulation pipeline.

These tests verify the complete workflow from object creation
through energy performance to economic analysis, without
requiring network access (no PVGIS calls).

Tests cover:
- Full prosumer pipeline (PV + BESS + Consumer -> energy -> economics)
- Full REC pipeline (Prosumers + Consumers -> energy -> economics)
- Output data consistency and shapes
- Numerical stability across a full day simulation
"""

import pytest
import numpy as np
from src.rec_sim.PvPanels import PvPanels
from src.rec_sim.Bess import Bess
from src.rec_sim.Consumer import Consumer
from src.rec_sim.Prosumer import Prosumer
from src.rec_sim.Rec import Rec
from tests.conftest import TIME_STEP_15MIN, N_STEPS_DAY


# ===========================================================================
# Helpers to build a complete simulation without PVGIS
# ===========================================================================
def _build_irradiance(n_steps=N_STEPS_DAY):
    """Generate synthetic irradiance for n_steps at 15-min resolution."""
    hours = np.arange(n_steps) * 0.25
    beam = np.maximum(0, 800 * np.sin((hours - 6) / 12 * np.pi))
    beam[hours < 6] = 0
    beam[hours > 18] = 0
    return {
        'I_beam': beam,
        'I_skydiff': beam * 0.15,
        'I_grounddiff': beam * 0.03,
        't_amb': 20 + 5 * np.sin((hours - 6) / 24 * 2 * np.pi),
        'wind_speed': np.ones(n_steps) * 2.0,
    }


def _build_pv(pv_id, n_series, n_parallel, irr):
    """Create and compute a PV system from synthetic irradiance."""
    pv = PvPanels(
        id=pv_id, cap_cost=1500, opex_cost=40,
        inc_year=900, inc_start_end=[1, 10], tax_year=0,
        n_series=n_series, n_parallel=n_parallel,
        dc_ac_efficiency=0.97, mismatch_loss=0.02,
        wiring_loss=0.015, soiling_loss=0.03,
    )
    pv.compute_output(
        slope=30, theta=None,
        I_beam=irr['I_beam'], I_skydiff=irr['I_skydiff'],
        I_grounddiff=irr['I_grounddiff'], t_amb=irr['t_amb'],
        wind_speed=irr['wind_speed'],
    )
    return pv


def _build_bess(bess_id, cap=2.56, soc_in=0.2, soc_max=0.8):
    """Create a BESS with realistic parameters."""
    return Bess(
        id=bess_id, cap=cap, c_rate=1.0,
        soc_in=soc_in, soc_max=soc_max, soc_min=0.2,
        cap_cost=720, opex_cost=20,
        inc_year=0, inc_start_end=[0, 0], tax_year=0,
        eta_charge=0.95, eta_discharge=0.95,
        self_discharge_rate_per_hour=0.00004,
        lifetime_years=15,
    )


def _build_consumer(cons_id, n_steps=N_STEPS_DAY, base_load=1.0):
    """Create a consumer with sinusoidal daily profile."""
    hours = np.arange(n_steps) * 0.25
    dem = base_load + 0.5 * np.sin(hours / 24 * 2 * np.pi - np.pi / 2)
    return Consumer(id=cons_id, dem={'electricity': dem})


# ===========================================================================
# Full prosumer pipeline
# ===========================================================================
class TestProsumerIntegration:

    @pytest.mark.integration
    def test_full_prosumer_pipeline(self):
        """Complete prosumer simulation: PV + BESS + Consumer -> energy -> economics."""
        irr = _build_irradiance()
        pv = _build_pv('pv1', 3, 2, irr)
        bess = _build_bess('bess1')
        cons = _build_consumer('c1')

        pros = Prosumer(
            id='pros1', carriers=['electricity'],
            systems=[pv], users=[cons], bess=[bess],
        )

        # Energy performance
        pros.energy_performance(time=TIME_STEP_15MIN)
        ep = pros.en_perf_evolution['electricity']

        # Check all expected keys exist
        expected_keys = ['prod', 'dem', 'self_cons', 'surplus', 'unmet',
                         'power_from_source', 'supply', 'soc',
                         'self_cons_without_bess', 'surplus_without_bess', 'unmet_without_bess']
        for key in expected_keys:
            assert key in ep, f"Missing key: {key}"

        # Check array lengths match
        for key in expected_keys:
            assert len(ep[key]) == N_STEPS_DAY, f"Wrong length for {key}"

        # Check no NaN values
        for key in expected_keys:
            assert not np.any(np.isnan(ep[key])), f"NaN in {key}"

        # Check no negative production or demand
        assert np.all(ep['prod'] >= 0)
        assert np.all(ep['dem'] >= 0)
        assert np.all(ep['self_cons'] >= -1e-10)

        # Economic performance
        flows = {
            'electricity': {
                'sold': float(np.sum(ep['surplus'])) / 1000 * TIME_STEP_15MIN,
                'self_cons': float(np.sum(ep['self_cons'])) / 1000 * TIME_STEP_15MIN,
                'purchased': float(np.sum(ep['unmet'])) / 1000 * TIME_STEP_15MIN,
                'price_sold': 104,
                'price_buy': 130,
                'decay': 0.005,
            }
        }
        ec = pros.economic_performance(
            time_horizon=20, tax_rate=0.2, int_rate=0.03,
            other_capex_perc=[0], en_flows_and_prices=flows,
        )
        assert 'NPV' in ec
        assert 'pbp' in ec
        assert 'capex' in ec
        assert ec['capex'] > 0
        # [FIX #3] OPEX must be non-zero
        assert ec['cost_opex'][1] > 0
        # [FIX #5] Purchased energy cost must be non-zero when there is unmet demand
        if float(np.sum(ep['unmet'])) > 0:
            assert ec['cost_resources'][1] > 0

    @pytest.mark.integration
    def test_prosumer_no_bess_pipeline(self):
        """Complete prosumer simulation without BESS."""
        irr = _build_irradiance()
        pv = _build_pv('pv1', 2, 2, irr)
        cons = _build_consumer('c1', base_load=2.0)

        pros = Prosumer(
            id='pros_nb', carriers=['electricity'],
            systems=[pv], users=[cons], bess=[],
        )
        pros.energy_performance(time=TIME_STEP_15MIN)
        ep = pros.en_perf_evolution['electricity']

        # Energy balance
        np.testing.assert_allclose(
            ep['self_cons'] + ep['surplus'], ep['prod'], atol=1e-6
        )
        np.testing.assert_allclose(
            ep['self_cons'] + ep['unmet'], ep['dem'], atol=1e-6
        )


# ===========================================================================
# Full REC pipeline
# ===========================================================================
class TestRecIntegration:

    @pytest.mark.integration
    def test_full_rec_pipeline(self):
        """Complete REC simulation: 2 prosumers + consumers -> energy -> economics."""
        irr = _build_irradiance()

        # Prosumer 1: PV + BESS
        pv1 = _build_pv('pv1', 3, 2, irr)
        bess1 = _build_bess('bess1')
        cons1 = _build_consumer('c1', base_load=1.0)
        pros1 = Prosumer(
            id='pros1', carriers=['electricity'],
            systems=[pv1], users=[cons1], bess=[bess1],
        )
        pros1.energy_performance(time=TIME_STEP_15MIN)

        # Prosumer 2: PV only
        pv2 = _build_pv('pv2', 2, 1, irr)
        cons2 = _build_consumer('c2', base_load=0.5)
        pros2 = Prosumer(
            id='pros2', carriers=['electricity'],
            systems=[pv2], users=[cons2], bess=[],
        )
        pros2.energy_performance(time=TIME_STEP_15MIN)

        # Additional consumers
        cons3 = _build_consumer('c3', base_load=1.5)
        cons4 = _build_consumer('c4', base_load=2.0)

        # REC with community PV
        pv_rec = _build_pv('pv_rec', 2, 2, irr)
        bess_rec = _build_bess('bess_rec', soc_in=0.3)

        rec = Rec(
            id='rec1', carriers=['electricity'],
            prosumers=[pros1, pros2],
            consumers=[cons3, cons4],
            rec_systems=[pv_rec],
            rec_bess=[bess_rec],
        )

        # Energy performance
        rec.energy_performance(time=TIME_STEP_15MIN)
        ep = rec.en_perf_evolution['electricity']

        # Check all expected keys
        expected_keys = ['prod', 'prod_net', 'prod_rec', 'dem', 'dem_net',
                         'shared', 'surplus_prosumers', 'selfcons_prosumers',
                         'unmet_prosumers', 'surplus', 'unmet',
                         'power_from_source', 'supply', 'soc']
        for key in expected_keys:
            assert key in ep, f"Missing key: {key}"

        # Check no NaN
        for key in expected_keys:
            assert not np.any(np.isnan(ep[key])), f"NaN in {key}"

        # Shared energy >= 0
        assert np.all(ep['shared'] >= -1e-10)

        # Economic performance
        flows = {
            'electricity': {
                'sold': float(np.sum(ep['prod_rec'])) / 1000 * TIME_STEP_15MIN,
                'self_cons': 0.0,  # La REC non ha carichi propri, nessun autoconsumo
                'purchased': 0.0,  # La REC non acquista energia dalla rete
                'price_sold': 104,
                'price_buy': 130,
                'decay': 0.005,
            }
        }
        ec = rec.economic_performance(
            time_horizon=20, tax_rate=0.2, int_rate=0.03,
            other_capex_perc=[0.02], en_flows_and_prices=flows,
        )
        assert 'NPV' in ec
        assert 'cost_bess_replacement' in ec
        assert len(ec['cost_bess_replacement']) == 21
        # La REC non ha costi di acquisto dalla rete
        assert ec['cost_resources'][1] == 0.0


# ===========================================================================
# Numerical stability
# ===========================================================================
class TestNumericalStability:

    @pytest.mark.integration
    def test_no_nan_in_pv_output(self):
        """PV output contains no NaN values under normal conditions."""
        irr = _build_irradiance()
        pv = _build_pv('pv', 5, 5, irr)
        prod = pv.en_perf_evolution['electricity']['prod']
        assert not np.any(np.isnan(prod))
        assert not np.any(np.isinf(prod))

    @pytest.mark.integration
    def test_no_nan_in_bess_output(self):
        """BESS output contains no NaN values under rapid charge/discharge."""
        bess = _build_bess('b', soc_in=0.5)
        # Rapid alternating charge/discharge
        for power in [5.0, -5.0, 10.0, -10.0, 0.0, 3.0, -8.0]:
            _, soc, power_from_source, supply, _, _, _ = \
                bess.energy_performance(power, TIME_STEP_15MIN)
            assert not np.isnan(soc)
            assert not np.isnan(power_from_source)
            assert not np.isnan(supply)
            assert 0 <= soc <= 1.0

    @pytest.mark.integration
    def test_soc_stays_bounded(self):
        """SOC never goes below soc_min or above soc_max under stress."""
        bess = _build_bess('b', soc_in=0.5, soc_max=0.8)
        # Extreme charge
        for _ in range(100):
            _, soc, _, _, _, _, _ = bess.energy_performance(100.0, TIME_STEP_15MIN)
            assert soc <= bess.soc_max + 1e-10
        # Extreme discharge
        for _ in range(100):
            _, soc, _, _, _, _, _ = bess.energy_performance(-100.0, TIME_STEP_15MIN)
            assert soc >= bess.soc_min - 1e-10
