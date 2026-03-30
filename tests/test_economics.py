"""
Unit tests for the Economics class.

Tests cover:
- CAPEX calculation
- Annual cashflow structure (revenues, costs)
- NPV and payback period calculation
- Incentive application within time windows
- Battery replacement cost scheduling
- Other costs/revenues within duration windows
"""

import pytest
import numpy as np
from src.rec_sim.Economics import Economics
from src.rec_sim.PvPanels import PvPanels
from src.rec_sim.Bess import Bess


# ===========================================================================
# Helpers
# ===========================================================================
def _make_pv(cap_cost=1500, inc_year=0, inc_start_end=None):
    """Create a minimal PV for economics testing."""
    if inc_start_end is None:
        inc_start_end = [0, 0]
    return PvPanels(
        id='pv', cap_cost=cap_cost, opex_cost=40,
        inc_year=inc_year, inc_start_end=inc_start_end, tax_year=0,
        n_series=1, n_parallel=1,
        dc_ac_efficiency=1.0, mismatch_loss=0.0,
        wiring_loss=0.0, soiling_loss=0.0,
    )


def _make_bess(cap_cost=720, lifetime=10, annual_fade=0.02,
               inc_year=0, inc_start_end=None):
    """Create a minimal BESS for economics testing."""
    if inc_start_end is None:
        inc_start_end = [0, 0]
    return Bess(
        id='bess', cap_module=2.56, v=25.6, i_max=100, i_min=5,
        soc_in=0.5, soc_max=0.8, soc_min=0.2,
        n_series=1, n_parallel=1,
        cap_cost=cap_cost, opex_cost=20,
        inc_year=inc_year, inc_start_end=inc_start_end, tax_year=0,
        lifetime_years=lifetime,
        annual_capacity_fade=annual_fade,
    )


def _make_flows():
    """Standard annual energy flows for testing."""
    return {
        'electricity': {
            'sold': 100,          # MWh
            'self_cons': 200,     # MWh
            'purchased': 10,      # MWh
            'price_sold': 100,    # €/MWh
            'price_buy': 130,     # €/MWh
            'decay': 0.02,
        }
    }


# ===========================================================================
# CAPEX
# ===========================================================================
class TestCapex:

    @pytest.mark.unit
    def test_capex_single_component(self):
        """CAPEX = cap_cost * cap for a single component."""
        pv = _make_pv(cap_cost=1500)
        flows = _make_flows()
        ec = Economics(components=[pv], annual_en_flows_and_prices=flows)
        result = ec.compute_cashflow(time_horizon=20, tax_rate=0.2, int_rate=0.03)
        expected_capex = 1500 * pv.cap
        assert result['capex'] == pytest.approx(expected_capex, rel=1e-4)

    @pytest.mark.unit
    def test_capex_with_other_percentage(self):
        """CAPEX is scaled by other_capex_perc: total = raw / (1 - perc)."""
        pv = _make_pv(cap_cost=1000)
        flows = _make_flows()
        ec = Economics(components=[pv], annual_en_flows_and_prices=flows)
        result = ec.compute_cashflow(
            time_horizon=20, tax_rate=0.2, int_rate=0.03,
            other_capex_perc=[0.1],
        )
        raw = 1000 * pv.cap
        expected = raw / (1 - 0.1)
        assert result['capex'] == pytest.approx(expected, rel=1e-4)

    @pytest.mark.unit
    def test_capex_multiple_components(self):
        """CAPEX sums across multiple components."""
        pv = _make_pv(cap_cost=1500)
        bess = _make_bess(cap_cost=720)
        flows = _make_flows()
        ec = Economics(components=[pv, bess], annual_en_flows_and_prices=flows)
        result = ec.compute_cashflow(time_horizon=20, tax_rate=0.2, int_rate=0.03)
        expected = (1500 * pv.cap + 720 * bess.cap)
        assert result['capex'] == pytest.approx(expected, rel=1e-4)


# ===========================================================================
# Cashflow structure
# ===========================================================================
class TestCashflow:

    @pytest.mark.unit
    def test_cashflow_length(self):
        """Cashflow arrays have time_horizon + 1 entries (year 0 to N)."""
        pv = _make_pv()
        flows = _make_flows()
        ec = Economics(components=[pv], annual_en_flows_and_prices=flows)
        result = ec.compute_cashflow(time_horizon=20, tax_rate=0.2, int_rate=0.03)
        assert len(result['rev_from_sale']) == 21
        assert len(result['cost_opex']) == 21

    @pytest.mark.unit
    def test_year_zero_no_revenue(self):
        """Year 0 has no revenues (only investment)."""
        pv = _make_pv()
        flows = _make_flows()
        ec = Economics(components=[pv], annual_en_flows_and_prices=flows)
        result = ec.compute_cashflow(time_horizon=10, tax_rate=0.2, int_rate=0.03)
        assert result['rev_from_sale'][0] == 0
        assert result['rev_savings'][0] == 0
        assert result['rev_incentives'][0] == 0

    @pytest.mark.unit
    def test_revenue_from_sale(self):
        """rev_from_sale = sold * price_sold * (1-decay)^(year-1)."""
        pv = _make_pv()
        flows = _make_flows()
        ec = Economics(components=[pv], annual_en_flows_and_prices=flows)
        result = ec.compute_cashflow(time_horizon=5, tax_rate=0.0, int_rate=0.03)
        # Year 1: 100 * 100 * (1-0.02)^0 = 10000
        assert result['rev_from_sale'][1] == pytest.approx(10000, rel=1e-4)
        # Year 2: 100 * 100 * (1-0.02)^1 = 9800
        assert result['rev_from_sale'][2] == pytest.approx(9800, rel=1e-4)

    @pytest.mark.unit
    def test_tax_on_sale(self):
        """cost_taxes_on_sale = rev_from_sale * tax_rate."""
        pv = _make_pv()
        flows = _make_flows()
        ec = Economics(components=[pv], annual_en_flows_and_prices=flows)
        result = ec.compute_cashflow(time_horizon=5, tax_rate=0.2, int_rate=0.03)
        for year in range(1, 6):
            assert result['cost_taxes_on_sale'][year] == pytest.approx(
                result['rev_from_sale'][year] * 0.2, rel=1e-6
            )


# ===========================================================================
# Incentives
# ===========================================================================
class TestIncentives:

    @pytest.mark.unit
    def test_incentives_within_window(self):
        """Incentives are applied only within inc_start_end range."""
        pv = _make_pv(inc_year=5000, inc_start_end=[2, 5])
        flows = _make_flows()
        ec = Economics(components=[pv], annual_en_flows_and_prices=flows)
        result = ec.compute_cashflow(time_horizon=10, tax_rate=0.2, int_rate=0.03)
        assert result['rev_incentives'][1] == pytest.approx(0)
        assert result['rev_incentives'][2] == pytest.approx(5000)
        assert result['rev_incentives'][5] == pytest.approx(5000)
        assert result['rev_incentives'][6] == pytest.approx(0)


# ===========================================================================
# NPV and PBP
# ===========================================================================
class TestNpvPbp:

    @pytest.mark.unit
    def test_npv_type(self):
        """NPV is a scalar float."""
        pv = _make_pv()
        flows = _make_flows()
        ec = Economics(components=[pv], annual_en_flows_and_prices=flows)
        result = ec.compute_cashflow(time_horizon=20, tax_rate=0.2, int_rate=0.03)
        assert isinstance(result['NPV'], (float, np.floating))

    @pytest.mark.unit
    def test_pbp_positive_for_profitable_project(self):
        """PBP is positive when project generates positive cashflow."""
        pv = _make_pv(cap_cost=100)  # Very cheap PV
        flows = _make_flows()
        ec = Economics(components=[pv], annual_en_flows_and_prices=flows)
        result = ec.compute_cashflow(time_horizon=20, tax_rate=0.0, int_rate=0.03)
        assert result['pbp'] > 0

    @pytest.mark.unit
    def test_higher_discount_rate_lower_npv(self):
        """Higher discount rate produces lower NPV."""
        pv = _make_pv()
        flows = _make_flows()
        ec = Economics(components=[pv], annual_en_flows_and_prices=flows)
        result_low = ec.compute_cashflow(time_horizon=20, tax_rate=0.2, int_rate=0.01)
        result_high = ec.compute_cashflow(time_horizon=20, tax_rate=0.2, int_rate=0.10)
        assert result_high['NPV'] < result_low['NPV']


# ===========================================================================
# Battery replacement
# ===========================================================================
class TestBatteryReplacement:

    @pytest.mark.unit
    def test_replacement_cost_at_lifetime(self):
        """Replacement cost appears at year = lifetime_years."""
        bess = _make_bess(cap_cost=720, lifetime=10)
        flows = _make_flows()
        ec = Economics(components=[bess], annual_en_flows_and_prices=flows)
        result = ec.compute_cashflow(time_horizon=20, tax_rate=0.2, int_rate=0.03)
        replacement = result['cost_bess_replacement']
        # Cost at year 10 and year 20
        assert replacement[10] == pytest.approx(720 * bess.cap, rel=1e-4)
        assert replacement[20] == pytest.approx(720 * bess.cap, rel=1e-4)
        # No replacement at other years
        assert replacement[5] == pytest.approx(0.0)
        assert replacement[15] == pytest.approx(0.0)

    @pytest.mark.unit
    def test_no_replacement_when_lifetime_exceeds_horizon(self):
        """No replacement cost when lifetime > time_horizon."""
        bess = _make_bess(cap_cost=720, lifetime=25)
        flows = _make_flows()
        ec = Economics(components=[bess], annual_en_flows_and_prices=flows)
        result = ec.compute_cashflow(time_horizon=20, tax_rate=0.2, int_rate=0.03)
        np.testing.assert_allclose(result['cost_bess_replacement'], 0.0)

    @pytest.mark.unit
    def test_replacement_increases_total_cost(self):
        """Battery replacement makes the project more expensive (lower NPV)."""
        bess_long = _make_bess(cap_cost=720, lifetime=25)  # no replacement in 20y
        bess_short = _make_bess(cap_cost=720, lifetime=10)  # replacement at y10, y20
        flows = _make_flows()

        ec_long = Economics(components=[bess_long], annual_en_flows_and_prices=flows)
        ec_short = Economics(components=[bess_short], annual_en_flows_and_prices=flows)

        result_long = ec_long.compute_cashflow(time_horizon=20, tax_rate=0.2, int_rate=0.03)
        result_short = ec_short.compute_cashflow(time_horizon=20, tax_rate=0.2, int_rate=0.03)

        assert result_short['NPV'] < result_long['NPV']

    @pytest.mark.unit
    def test_pv_has_no_replacement(self):
        """PV panels (no lifetime_years attribute by default) have no replacement cost."""
        pv = _make_pv()
        flows = _make_flows()
        ec = Economics(components=[pv], annual_en_flows_and_prices=flows)
        result = ec.compute_cashflow(time_horizon=20, tax_rate=0.2, int_rate=0.03)
        np.testing.assert_allclose(result['cost_bess_replacement'], 0.0)
