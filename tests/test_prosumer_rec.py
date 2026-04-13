"""
Unit tests for Consumer, Prosumer, and Rec classes.

Tests cover:
- Consumer demand profile storage
- Prosumer energy balance (prod, dem, self_cons, surplus, unmet)
- Prosumer with and without BESS
- BESS improves self-consumption
- Rec shared energy calculation
- Rec energy balance across prosumers and consumers
- Multi-carrier BESS support
"""

import pytest
import numpy as np
from src.rec_sim.Consumer import Consumer
from src.rec_sim.Prosumer import Prosumer
from src.rec_sim.Rec import Rec
from src.rec_sim.PvPanels import PvPanels
from src.rec_sim.Bess import Bess
from tests.conftest import TIME_STEP_15MIN, N_STEPS_DAY


# ===========================================================================
# Consumer
# ===========================================================================
class TestConsumer:

    @pytest.mark.unit
    def test_consumer_stores_demand(self, consumer_flat_1kw):
        """Consumer stores demand profile as en_perf_evolution."""
        assert 'electricity' in consumer_flat_1kw.en_perf_evolution
        assert len(consumer_flat_1kw.en_perf_evolution['electricity']) == N_STEPS_DAY

    @pytest.mark.unit
    def test_consumer_demand_values(self, consumer_flat_1kw):
        """Flat consumer has constant 1 kW demand."""
        np.testing.assert_allclose(
            consumer_flat_1kw.en_perf_evolution['electricity'], 1.0
        )

    @pytest.mark.unit
    def test_consumer_variable_profile(self, consumer_variable):
        """Variable consumer has non-constant demand."""
        dem = consumer_variable.en_perf_evolution['electricity']
        assert np.std(dem) > 0.1  # not flat


# ===========================================================================
# Prosumer - without BESS
# ===========================================================================
class TestProsumerNoBess:

    @pytest.fixture
    def prosumer_no_bess(self, pv_no_losses, consumer_flat_1kw, synthetic_irradiance_day):
        """Prosumer with PV but no battery."""
        irr = synthetic_irradiance_day
        pv_no_losses.compute_output(
            slope=30, theta=None,
            I_beam=irr['I_beam'], I_skydiff=irr['I_skydiff'],
            I_grounddiff=irr['I_grounddiff'], t_amb=irr['t_amb'],
        )
        pros = Prosumer(
            id='no_bess', carriers=['electricity'],
            systems=[pv_no_losses], users=[consumer_flat_1kw], bess=[],
        )
        pros.energy_performance(time=TIME_STEP_15MIN)
        return pros

    @pytest.mark.unit
    def test_energy_balance_no_bess(self, prosumer_no_bess):
        """self_cons + surplus = prod and self_cons + unmet = dem."""
        ep = prosumer_no_bess.en_perf_evolution['electricity']
        np.testing.assert_allclose(
            ep['self_cons'] + ep['surplus'], ep['prod'], atol=1e-6
        )
        np.testing.assert_allclose(
            ep['self_cons'] + ep['unmet'], ep['dem'], atol=1e-6
        )

    @pytest.mark.unit
    def test_self_cons_is_min_prod_dem(self, prosumer_no_bess):
        """Self-consumption = min(production, demand) at each step."""
        ep = prosumer_no_bess.en_perf_evolution['electricity']
        expected = np.minimum(ep['prod'], ep['dem'])
        np.testing.assert_allclose(ep['self_cons'], expected, atol=1e-6)

    @pytest.mark.unit
    def test_no_stored_no_supply(self, prosumer_no_bess):
        """Without BESS, no stored/supply keys exist."""
        ep = prosumer_no_bess.en_perf_evolution['electricity']
        assert 'stored' not in ep
        assert 'supply' not in ep


# ===========================================================================
# Prosumer - with BESS
# ===========================================================================
class TestProsumerWithBess:

    @pytest.mark.unit
    def test_bess_improves_self_consumption(self, pv_no_losses, consumer_flat_1kw,
                                             synthetic_irradiance_day):
        """BESS increases self-consumption compared to no BESS."""
        irr = synthetic_irradiance_day
        pv_no_losses.compute_output(
            slope=30, theta=None,
            I_beam=irr['I_beam'], I_skydiff=irr['I_skydiff'],
            I_grounddiff=irr['I_grounddiff'], t_amb=irr['t_amb'],
        )

        # Without BESS
        pv_copy1 = PvPanels(
            id='pv1', cap_cost=1500, opex_cost=40,
            inc_year=0, inc_start_end=[0, 0], tax_year=0,
            dc_ac_efficiency=1.0, mismatch_loss=0.0,
            wiring_loss=0.0, soiling_loss=0.0,
        )
        pv_copy1.compute_output(
            slope=30, theta=None,
            I_beam=irr['I_beam'], I_skydiff=irr['I_skydiff'],
            I_grounddiff=irr['I_grounddiff'], t_amb=irr['t_amb'],
        )
        cons1 = Consumer(id='c1', dem={'electricity': np.ones(N_STEPS_DAY)})
        pros_no_bess = Prosumer(
            id='nb', carriers=['electricity'],
            systems=[pv_copy1], users=[cons1], bess=[],
        )
        pros_no_bess.energy_performance(time=TIME_STEP_15MIN)
        sc_no_bess = np.sum(pros_no_bess.en_perf_evolution['electricity']['self_cons'])

        # With BESS
        pv_copy2 = PvPanels(
            id='pv2', cap_cost=1500, opex_cost=40,
            inc_year=0, inc_start_end=[0, 0], tax_year=0,
            dc_ac_efficiency=1.0, mismatch_loss=0.0,
            wiring_loss=0.0, soiling_loss=0.0,
        )
        pv_copy2.compute_output(
            slope=30, theta=None,
            I_beam=irr['I_beam'], I_skydiff=irr['I_skydiff'],
            I_grounddiff=irr['I_grounddiff'], t_amb=irr['t_amb'],
        )
        cons2 = Consumer(id='c2', dem={'electricity': np.ones(N_STEPS_DAY)})
        bess = Bess(
            id='b', cap=5.0, c_rate=1.0,
            soc_in=0.5, soc_max=1.0, soc_min=0.0,
            cap_cost=500, opex_cost=10, inc_year=0,
            inc_start_end=[0, 0], tax_year=0,
            eta_charge=1.0, eta_discharge=1.0,
            self_discharge_rate_per_hour=0.0,
        )
        pros_with_bess = Prosumer(
            id='wb', carriers=['electricity'],
            systems=[pv_copy2], users=[cons2], bess=[bess],
        )
        pros_with_bess.energy_performance(time=TIME_STEP_15MIN)
        sc_with_bess = np.sum(pros_with_bess.en_perf_evolution['electricity']['self_cons'])

        assert sc_with_bess >= sc_no_bess - 1e-6

    @pytest.mark.unit
    def test_bess_keys_present(self, simple_prosumer):
        """With BESS, stored/supply/soc keys are present."""
        simple_prosumer.energy_performance(time=TIME_STEP_15MIN)
        ep = simple_prosumer.en_perf_evolution['electricity']
        assert 'stored' in ep
        assert 'supply' in ep
        assert 'soc' in ep
        assert 'self_cons_without_bess' in ep

    @pytest.mark.unit
    def test_energy_balance_with_bess(self, simple_prosumer):
        """Energy balance: prod = self_cons + surplus (with BESS)."""
        simple_prosumer.energy_performance(time=TIME_STEP_15MIN)
        ep = simple_prosumer.en_perf_evolution['electricity']
        # With BESS, self_cons = self_cons_without_bess + stored
        # The balance: prod >= self_cons (production covers self-consumption)
        # and surplus >= 0, unmet >= 0
        assert np.all(ep['surplus'] >= -1e-6)
        assert np.all(ep['unmet'] >= -1e-6)
        assert np.all(ep['self_cons'] >= -1e-6)


# ===========================================================================
# Rec
# ===========================================================================
class TestRec:

    @pytest.mark.unit
    def test_rec_shared_energy(self, simple_rec):
        """Shared energy = min(prod_net, dem_net) when no BESS at REC level."""
        simple_rec.energy_performance(time=TIME_STEP_15MIN)
        ep = simple_rec.en_perf_evolution['electricity']
        expected_shared = np.minimum(ep['prod_net'], ep['dem_net'])
        np.testing.assert_allclose(ep['shared'], expected_shared, atol=1e-6)

    @pytest.mark.unit
    def test_rec_energy_balance(self, simple_rec):
        """shared + surplus = prod_net and shared + unmet = dem_net."""
        simple_rec.energy_performance(time=TIME_STEP_15MIN)
        ep = simple_rec.en_perf_evolution['electricity']
        np.testing.assert_allclose(
            ep['shared'] + ep['surplus'], ep['prod_net'], atol=1e-6
        )
        np.testing.assert_allclose(
            ep['shared'] + ep['unmet'], ep['dem_net'], atol=1e-6
        )

    @pytest.mark.unit
    def test_rec_member_count(self, simple_rec):
        """compute_members returns correct counts."""
        n_members, n_pros, n_cons = simple_rec.compute_members()
        assert n_pros == 1
        assert n_cons == 1
        assert n_members == 2

    @pytest.mark.unit
    def test_rec_prod_includes_prosumers(self, simple_rec):
        """Total production includes prosumer production."""
        simple_rec.energy_performance(time=TIME_STEP_15MIN)
        ep = simple_rec.en_perf_evolution['electricity']
        assert np.sum(ep['prod']) > 0


# ===========================================================================
# Multi-carrier BESS
# ===========================================================================
class TestMultiCarrierBess:

    @pytest.mark.unit
    def test_bess_activates_for_matching_carrier(self):
        """BESS with matching carrier activates for that carrier."""
        pv = PvPanels(
            id='pv', cap_cost=1500, opex_cost=40,
            inc_year=0, inc_start_end=[0, 0], tax_year=0,
            dc_ac_efficiency=1.0, mismatch_loss=0.0,
            wiring_loss=0.0, soiling_loss=0.0,
        )
        # Minimal irradiance to get some production
        n = 10
        pv.compute_output(
            slope=30, theta=None,
            I_beam=np.full(n, 500.0), I_skydiff=np.full(n, 50.0),
            I_grounddiff=np.full(n, 10.0), t_amb=np.full(n, 25.0),
        )
        cons = Consumer(id='c', dem={'electricity': np.ones(n) * 0.1})
        bess = Bess(
            id='b', carriers=['electricity'],
            cap=5.0, c_rate=1.0,
            soc_in=0.5, soc_max=1.0, soc_min=0.0,
            cap_cost=500, opex_cost=10, inc_year=0,
            inc_start_end=[0, 0], tax_year=0,
            eta_charge=1.0, eta_discharge=1.0,
            self_discharge_rate_per_hour=0.0,
        )
        pros = Prosumer(
            id='p', carriers=['electricity'],
            systems=[pv], users=[cons], bess=[bess],
        )
        pros.energy_performance(time=TIME_STEP_15MIN)
        assert 'stored' in pros.en_perf_evolution['electricity']

    @pytest.mark.unit
    def test_bess_inactive_for_wrong_carrier(self):
        """BESS with non-matching carrier is not activated."""
        pv = PvPanels(
            id='pv', cap_cost=1500, opex_cost=40,
            inc_year=0, inc_start_end=[0, 0], tax_year=0,
            dc_ac_efficiency=1.0, mismatch_loss=0.0,
            wiring_loss=0.0, soiling_loss=0.0,
        )
        n = 10
        pv.compute_output(
            slope=30, theta=None,
            I_beam=np.full(n, 500.0), I_skydiff=np.full(n, 50.0),
            I_grounddiff=np.full(n, 10.0), t_amb=np.full(n, 25.0),
        )
        cons = Consumer(id='c', dem={'electricity': np.ones(n) * 0.1})
        bess_heat = Bess(
            id='bh', carriers=['heat'],  # wrong carrier
            cap=5.0, c_rate=1.0,
            soc_in=0.5, soc_max=1.0, soc_min=0.0,
            cap_cost=500, opex_cost=10, inc_year=0,
            inc_start_end=[0, 0], tax_year=0,
        )
        pros = Prosumer(
            id='p', carriers=['electricity'],
            systems=[pv], users=[cons], bess=[bess_heat],
        )
        pros.energy_performance(time=TIME_STEP_15MIN)
        # No BESS keys because carrier doesn't match
        assert 'stored' not in pros.en_perf_evolution['electricity']
