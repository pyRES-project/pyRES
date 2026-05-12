"""
Unit tests for the Economics class.

Tests cover:
- CAPEX calculation
- Annual cashflow structure (revenues, costs)
- NPV and payback period calculation
- Incentive application within time windows
- Battery replacement cost scheduling
- [FIX #3] OPEX and tax are non-zero in cashflow
- [FIX #2] Price decay applies to purchase costs
- [FIX #4] Production degradation reduces energy flows
- [FIX #5] Purchased energy is correctly accounted
"""

import pytest
import numpy as np
from src.rec_sim.Economics import Economics
from src.rec_sim.PvPanels import PvPanels
from src.rec_sim.Bess import Bess


# ===========================================================================
# Helpers
# ===========================================================================
def _make_pv(cap_cost=1500, opex_cost=40, inc_year=0, inc_start_end=None, tax_year=0):
    if inc_start_end is None:
        inc_start_end = [0, 0]
    return PvPanels(
        id='pv', cap_cost=cap_cost, opex_cost=opex_cost,
        inc_year=inc_year, inc_start_end=inc_start_end, tax_year=tax_year,
        n_series=1, n_parallel=1,
        dc_ac_efficiency=1.0, mismatch_loss=0.0,
        wiring_loss=0.0, soiling_loss=0.0,
    )


def _make_bess(cap_cost=720, lifetime=10,
               inc_year=0, inc_start_end=None, opex_cost=20, tax_year=0):
    if inc_start_end is None:
        inc_start_end = [0, 0]
    return Bess(
        id='bess', cap=2.56, c_rate=1.0,
        soc_in=0.5, soc_max=0.8, soc_min=0.2,
        cap_cost=cap_cost, opex_cost=opex_cost,
        inc_year=inc_year, inc_start_end=inc_start_end, tax_year=tax_year,
        lifetime_years=lifetime,
    )


def _make_flows(**overrides):
    defaults = {
        'electricity': {
            'sold': 100,
            'self_cons': 200,
            'purchased': 50,
            'price_sold': 100,
            'price_buy': 130,
            'decay': 0.005,
        }
    }
    defaults['electricity'].update(overrides)
    return defaults


# ===========================================================================
# CAPEX
# ===========================================================================
class TestCapex:

    @pytest.mark.unit
    def test_capex_single_component(self):
        pv = _make_pv(cap_cost=1500)
        flows = _make_flows()
        ec = Economics(components=[pv], en_flows_and_prices=flows)
        result = ec.compute_cashflow(time_horizon=20, tax_rate=0.2, int_rate=0.03)
        expected_capex = 1500 * pv.cap
        assert result['capex'] == pytest.approx(expected_capex, rel=1e-4)

    @pytest.mark.unit
    def test_capex_with_other_percentage(self):
        pv = _make_pv(cap_cost=1000)
        flows = _make_flows()
        ec = Economics(components=[pv], en_flows_and_prices=flows)
        result = ec.compute_cashflow(time_horizon=20, tax_rate=0.2, int_rate=0.03, other_capex_perc=[0.1])
        raw = 1000 * pv.cap
        expected = raw / (1 - 0.1)
        assert result['capex'] == pytest.approx(expected, rel=1e-4)

    @pytest.mark.unit
    def test_capex_multiple_components(self):
        pv = _make_pv(cap_cost=1500)
        bess = _make_bess(cap_cost=720)
        flows = _make_flows()
        ec = Economics(components=[pv, bess], en_flows_and_prices=flows)
        result = ec.compute_cashflow(time_horizon=20, tax_rate=0.2, int_rate=0.03)
        expected = (1500 * pv.cap + 720 * bess.cap)
        assert result['capex'] == pytest.approx(expected, rel=1e-4)


# ===========================================================================
# Cashflow structure
# ===========================================================================
class TestCashflow:

    @pytest.mark.unit
    def test_cashflow_length(self):
        pv = _make_pv()
        flows = _make_flows()
        ec = Economics(components=[pv], en_flows_and_prices=flows)
        result = ec.compute_cashflow(time_horizon=20, tax_rate=0.2, int_rate=0.03)
        assert len(result['rev_from_sale']) == 21
        assert len(result['cost_opex']) == 21

    @pytest.mark.unit
    def test_year_zero_no_revenue(self):
        pv = _make_pv()
        flows = _make_flows()
        ec = Economics(components=[pv], en_flows_and_prices=flows)
        result = ec.compute_cashflow(time_horizon=10, tax_rate=0.2, int_rate=0.03)
        assert result['rev_from_sale'][0] == 0
        assert result['rev_savings'][0] == 0
        assert result['rev_incentives'][0] == 0

    @pytest.mark.unit
    def test_revenue_from_sale(self):
        pv = _make_pv()
        flows = _make_flows()
        ec = Economics(components=[pv], en_flows_and_prices=flows)
        result = ec.compute_cashflow(time_horizon=5, tax_rate=0.0, int_rate=0.03)
        assert result['rev_from_sale'][1] == pytest.approx(10000, rel=1e-4)
        assert result['rev_from_sale'][2] == pytest.approx(9950, rel=1e-4)

    @pytest.mark.unit
    def test_tax_on_sale(self):
        pv = _make_pv()
        flows = _make_flows()
        ec = Economics(components=[pv], en_flows_and_prices=flows)
        result = ec.compute_cashflow(time_horizon=5, tax_rate=0.2, int_rate=0.03)
        for year in range(1, 6):
            assert result['cost_taxes_on_sale'][year] == pytest.approx(
                result['rev_from_sale'][year] * 0.2, rel=1e-6
            )


# ===========================================================================
# [FIX #3] OPEX and tax non-zero
# ===========================================================================
class TestOpexAndTax:

    @pytest.mark.unit
    def test_opex_is_nonzero(self):
        """[FIX #3] OPEX must be > 0 when opex_cost > 0."""
        pv = _make_pv(opex_cost=40)
        flows = _make_flows()
        ec = Economics(components=[pv], en_flows_and_prices=flows)
        result = ec.compute_cashflow(time_horizon=5, tax_rate=0.0, int_rate=0.03)
        expected_opex = 40 * pv.opex  # opex_cost × opex (capacity for O&M)
        assert result['cost_opex'][1] == pytest.approx(expected_opex, rel=1e-4)
        assert result['cost_opex'][1] > 0

    @pytest.mark.unit
    def test_tax_is_nonzero(self):
        """[FIX #3] Annual tax must be > 0 when tax_year > 0."""
        pv = _make_pv(tax_year=500)
        flows = _make_flows()
        ec = Economics(components=[pv], en_flows_and_prices=flows)
        result = ec.compute_cashflow(time_horizon=5, tax_rate=0.0, int_rate=0.03)
        assert result['cost_taxes'][1] == pytest.approx(500, rel=1e-4)
        assert result['cost_taxes'][1] > 0

    @pytest.mark.unit
    def test_opex_constant_across_years(self):
        """OPEX is constant across all years."""
        pv = _make_pv(opex_cost=40)
        flows = _make_flows()
        ec = Economics(components=[pv], en_flows_and_prices=flows)
        result = ec.compute_cashflow(time_horizon=10, tax_rate=0.0, int_rate=0.03)
        for year in range(1, 11):
            assert result['cost_opex'][year] == pytest.approx(result['cost_opex'][1], rel=1e-6)

    @pytest.mark.unit
    def test_opex_sum_multiple_components(self):
        """OPEX sums across multiple components."""
        pv = _make_pv(opex_cost=40)
        bess = _make_bess(opex_cost=20)
        flows = _make_flows()
        ec = Economics(components=[pv, bess], en_flows_and_prices=flows)
        result = ec.compute_cashflow(time_horizon=5, tax_rate=0.0, int_rate=0.03)
        expected = 40 * pv.opex + 20 * bess.opex
        assert result['cost_opex'][1] == pytest.approx(expected, rel=1e-4)


# ===========================================================================
# Purchase costs
# ===========================================================================
class TestPurchaseCost:

    @pytest.mark.unit
    def test_purchase_cost_constant(self):
        """Purchase cost = purchased * price_buy, constant across years."""
        pv = _make_pv()
        flows = _make_flows(purchased=50, price_buy=130, decay=0.05)
        ec = Economics(components=[pv], en_flows_and_prices=flows)
        result = ec.compute_cashflow(time_horizon=5, tax_rate=0.0, int_rate=0.03)
        for year in range(1, 6):
            assert result['cost_resources'][year] == pytest.approx(6500, rel=1e-4)

    @pytest.mark.unit
    def test_purchase_cost_zero_when_no_purchased(self):
        """No purchase cost when purchased = 0."""
        pv = _make_pv()
        flows = _make_flows(purchased=0)
        ec = Economics(components=[pv], en_flows_and_prices=flows)
        result = ec.compute_cashflow(time_horizon=5, tax_rate=0.0, int_rate=0.03)
        for year in range(1, 6):
            assert result['cost_resources'][year] == pytest.approx(0, abs=1e-6)


# ===========================================================================
# [FIX #4] Production degradation
# ===========================================================================
class TestProductionDegradation:

    @pytest.mark.unit
    def test_sale_revenue_decreases_with_degradation(self):
        """Sale revenue decreases due to production decay."""
        pv = _make_pv()
        flows = _make_flows(sold=100, price_sold=100, decay=0.01)
        ec = Economics(components=[pv], en_flows_and_prices=flows)
        result = ec.compute_cashflow(time_horizon=5, tax_rate=0.0, int_rate=0.0)
        # Year 1: 100 * 100 * (1-0.01)^0 = 10000
        assert result['rev_from_sale'][1] == pytest.approx(10000, rel=1e-4)
        # Year 2: 100 * (1-0.01)^1 * 100 = 9900
        assert result['rev_from_sale'][2] == pytest.approx(9900, rel=1e-4)
        # Year 5: 100 * (1-0.01)^4 * 100 = 9606 (approx)
        assert result['rev_from_sale'][5] == pytest.approx(100 * 0.99**4 * 100, rel=1e-3)

    @pytest.mark.unit
    def test_no_degradation_when_zero(self):
        """With decay=0, revenues stay constant."""
        pv = _make_pv()
        flows = _make_flows(sold=100, price_sold=100, decay=0.0)
        ec = Economics(components=[pv], en_flows_and_prices=flows)
        result = ec.compute_cashflow(time_horizon=5, tax_rate=0.0, int_rate=0.0)
        # Constant across years
        for year in range(1, 6):
            assert result['rev_from_sale'][year] == pytest.approx(10000, rel=1e-4)

    @pytest.mark.unit
    def test_purchased_cost_constant_with_degradation(self):
        """Purchased cost stays constant even when production degrades."""
        pv = _make_pv()
        flows = _make_flows(sold=100, self_cons=200, purchased=50,
                            price_buy=130, decay=0.1)
        ec = Economics(components=[pv], en_flows_and_prices=flows)
        result = ec.compute_cashflow(time_horizon=3, tax_rate=0.0, int_rate=0.0)
        for year in range(1, 4):
            assert result['cost_resources'][year] == pytest.approx(50 * 130, rel=1e-3)


# ===========================================================================
# Incentives
# ===========================================================================
class TestIncentives:

    @pytest.mark.unit
    def test_incentives_within_window(self):
        pv = _make_pv(inc_year=5000, inc_start_end=[2, 5])
        flows = _make_flows()
        ec = Economics(components=[pv], en_flows_and_prices=flows)
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
        pv = _make_pv()
        flows = _make_flows()
        ec = Economics(components=[pv], en_flows_and_prices=flows)
        result = ec.compute_cashflow(time_horizon=20, tax_rate=0.2, int_rate=0.03)
        assert isinstance(result['NPV'], (float, np.floating))

    @pytest.mark.unit
    def test_pbp_positive_for_profitable_project(self):
        pv = _make_pv(cap_cost=100)
        flows = _make_flows()
        ec = Economics(components=[pv], en_flows_and_prices=flows)
        result = ec.compute_cashflow(time_horizon=20, tax_rate=0.0, int_rate=0.03)
        assert result['pbp'] > 0

    @pytest.mark.unit
    def test_higher_discount_rate_lower_npv(self):
        pv = _make_pv()
        flows = _make_flows()
        ec = Economics(components=[pv], en_flows_and_prices=flows)
        result_low = ec.compute_cashflow(time_horizon=20, tax_rate=0.2, int_rate=0.01)
        result_high = ec.compute_cashflow(time_horizon=20, tax_rate=0.2, int_rate=0.10)
        assert result_high['NPV'] < result_low['NPV']


# ===========================================================================
# Battery replacement
# ===========================================================================
class TestBatteryReplacement:

    @pytest.mark.unit
    def test_replacement_cost_at_lifetime(self):
        bess = _make_bess(cap_cost=720, lifetime=10)
        flows = _make_flows()
        ec = Economics(components=[bess], en_flows_and_prices=flows)
        result = ec.compute_cashflow(time_horizon=20, tax_rate=0.2, int_rate=0.03)
        replacement = result['cost_bess_replacement']
        assert replacement[10] == pytest.approx(720 * bess.cap, rel=1e-4)
        assert replacement[20] == pytest.approx(720 * bess.cap, rel=1e-4)
        assert replacement[5] == pytest.approx(0.0)

    @pytest.mark.unit
    def test_no_replacement_when_lifetime_exceeds_horizon(self):
        bess = _make_bess(cap_cost=720, lifetime=25)
        flows = _make_flows()
        ec = Economics(components=[bess], en_flows_and_prices=flows)
        result = ec.compute_cashflow(time_horizon=20, tax_rate=0.2, int_rate=0.03)
        np.testing.assert_allclose(result['cost_bess_replacement'], 0.0)

    @pytest.mark.unit
    def test_replacement_increases_total_cost(self):
        bess_long = _make_bess(cap_cost=720, lifetime=25)
        bess_short = _make_bess(cap_cost=720, lifetime=10)
        flows = _make_flows()
        ec_long = Economics(components=[bess_long], en_flows_and_prices=flows)
        ec_short = Economics(components=[bess_short], en_flows_and_prices=flows)
        result_long = ec_long.compute_cashflow(time_horizon=20, tax_rate=0.2, int_rate=0.03)
        result_short = ec_short.compute_cashflow(time_horizon=20, tax_rate=0.2, int_rate=0.03)
        assert result_short['NPV'] < result_long['NPV']

    @pytest.mark.unit
    def test_pv_has_no_replacement(self):
        pv = _make_pv()
        flows = _make_flows()
        ec = Economics(components=[pv], en_flows_and_prices=flows)
        result = ec.compute_cashflow(time_horizon=20, tax_rate=0.2, int_rate=0.03)
        np.testing.assert_allclose(result['cost_bess_replacement'], 0.0)
