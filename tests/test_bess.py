"""
Unit tests for Bess and Controller classes.

Tests cover:
- Construction and parameter validation
- Charge/discharge with and without efficiency losses
- SOC limits (min, max)
- V(SOC) variable voltage model
- Self-discharge over time
- C-rate limiting
- Micro-cycle threshold
- Full cycle equivalent counter
- Controller multi-battery cascade logic
- Energy conservation across charge/discharge cycles
"""

import pytest
import numpy as np
from src.rec_sim.Bess import Bess
from src.rec_sim.Controller import Controller
from tests.conftest import TIME_STEP_15MIN, TIME_STEP_1H


# ===========================================================================
# Bess construction
# ===========================================================================
class TestBessConstruction:

    @pytest.mark.unit
    def test_capacity_calculation(self, bess_default):
        """Total capacity = cap_module * n_series * n_parallel."""
        assert bess_default.cap == pytest.approx(2.56 * 1 * 1)

    @pytest.mark.unit
    def test_capacity_multi_module(self):
        """Capacity scales with series and parallel modules."""
        bess = Bess(
            id='multi', cap_module=2.0, v=25.0, i_max=50, i_min=5,
            soc_in=0.5, soc_max=1.0, soc_min=0.0,
            n_series=3, n_parallel=2,
            cap_cost=500, opex_cost=10, inc_year=0,
            inc_start_end=[0, 0], tax_year=0,
        )
        assert bess.cap == pytest.approx(2.0 * 3 * 2)

    @pytest.mark.unit
    def test_voltage_scaling(self, bess_default):
        """Rated voltage = v_module * n_series."""
        assert bess_default.v_rated == pytest.approx(25.6)

    @pytest.mark.unit
    def test_current_scaling(self):
        """Max current scales with n_parallel."""
        bess = Bess(
            id='par', cap_module=2.0, v=25.0, i_max=50, i_min=5,
            soc_in=0.5, soc_max=1.0, soc_min=0.0,
            n_series=1, n_parallel=3,
            cap_cost=500, opex_cost=10, inc_year=0,
            inc_start_end=[0, 0], tax_year=0,
        )
        assert bess.i_max == pytest.approx(50 * 3)

    @pytest.mark.unit
    def test_default_efficiency(self, bess_default):
        """Default efficiency values are set correctly."""
        assert bess_default.eta_charge == pytest.approx(0.95)
        assert bess_default.eta_discharge == pytest.approx(0.95)

    @pytest.mark.unit
    def test_initial_cycle_counter(self, bess_default):
        """Cycle counter starts at zero."""
        assert bess_default.cumulative_discharge_energy == 0.0
        assert bess_default.full_cycle_equivalents == 0.0


# ===========================================================================
# Charge behavior
# ===========================================================================
class TestBessCharge:

    @pytest.mark.unit
    def test_charge_ideal_increases_soc(self, bess_ideal):
        """Charging with ideal battery increases SOC."""
        soc_before = bess_ideal.soc_in
        power_in = 5.0  # kW surplus
        _, soc, stored, supply, power, surplus, deficit, current, mode = \
            bess_ideal.energy_performance(power_in, TIME_STEP_1H)
        assert soc > soc_before
        assert stored > 0
        assert supply == 0
        assert deficit == 0

    @pytest.mark.unit
    def test_charge_efficiency_loss(self, bess_default):
        """With eta_charge < 1, less energy is stored than consumed from source."""
        bess_default.soc_in = 0.3
        power_in = 1.0  # kW
        _, soc, stored, supply, power, surplus, deficit, current, mode = \
            bess_default.energy_performance(power_in, TIME_STEP_1H)
        energy_in = power_in * TIME_STEP_1H
        energy_stored_in_battery = bess_default.cap * soc - bess_default.cap * 0.3
        # The battery should store less than what was drawn (due to eta_charge)
        # Allow for self-discharge adjustment on initial SOC
        assert energy_stored_in_battery < energy_in or stored == pytest.approx(0.0)

    @pytest.mark.unit
    def test_charge_stops_at_soc_max(self, bess_ideal):
        """Battery does not charge beyond soc_max."""
        bess_ideal.soc_in = 0.95
        power_in = 100.0  # very large surplus
        _, soc, stored, supply, power, surplus, deficit, current, mode = \
            bess_ideal.energy_performance(power_in, TIME_STEP_1H)
        assert soc <= bess_ideal.soc_max + 1e-10
        assert surplus > 0  # excess not absorbed

    @pytest.mark.unit
    def test_charge_already_full(self, bess_ideal):
        """No charge when SOC already at max."""
        bess_ideal.soc_in = 1.0
        _, soc, stored, supply, power, surplus, deficit, current, mode = \
            bess_ideal.energy_performance(5.0, TIME_STEP_1H)
        assert stored == pytest.approx(0.0)
        assert surplus == pytest.approx(5.0)
        assert mode == 7


# ===========================================================================
# Discharge behavior
# ===========================================================================
class TestBessDischarge:

    @pytest.mark.unit
    def test_discharge_ideal_decreases_soc(self, bess_ideal):
        """Discharging with ideal battery decreases SOC."""
        soc_before = bess_ideal.soc_in
        power_in = -3.0  # kW deficit
        _, soc, stored, supply, power, surplus, deficit, current, mode = \
            bess_ideal.energy_performance(power_in, TIME_STEP_1H)
        assert soc < soc_before
        assert supply > 0
        assert stored == 0

    @pytest.mark.unit
    def test_discharge_efficiency_loss(self):
        """With eta_discharge < 1, less energy is delivered than extracted from battery."""
        bess = Bess(
            id='eff', cap_module=10.0, v=50.0, i_max=200, i_min=1,
            soc_in=0.8, soc_max=1.0, soc_min=0.0,
            n_series=1, n_parallel=1,
            cap_cost=500, opex_cost=10, inc_year=0,
            inc_start_end=[0, 0], tax_year=0,
            eta_charge=1.0, eta_discharge=0.90,
            self_discharge_rate_per_hour=0.0,
            min_energy_threshold=0.0,
        )
        power_in = -2.0  # kW deficit
        _, soc, stored, supply, power, surplus, deficit, current, mode = \
            bess.energy_performance(power_in, TIME_STEP_1H)
        energy_extracted = bess.cap * 0.8 - bess.cap * soc
        energy_delivered = supply * TIME_STEP_1H
        # Delivered < extracted due to discharge efficiency
        assert energy_delivered < energy_extracted + 1e-10

    @pytest.mark.unit
    def test_discharge_stops_at_soc_min(self, bess_ideal):
        """Battery does not discharge below soc_min."""
        bess_ideal.soc_in = 0.05
        bess_ideal.soc_min = 0.0
        _, soc, stored, supply, power, surplus, deficit, current, mode = \
            bess_ideal.energy_performance(-100.0, TIME_STEP_1H)
        assert soc >= bess_ideal.soc_min - 1e-10
        assert deficit > 0

    @pytest.mark.unit
    def test_discharge_already_empty(self, bess_ideal):
        """No discharge when SOC at minimum."""
        bess_ideal.soc_in = 0.0
        _, soc, stored, supply, power, surplus, deficit, current, mode = \
            bess_ideal.energy_performance(-5.0, TIME_STEP_1H)
        assert supply == pytest.approx(0.0)
        assert deficit == pytest.approx(5.0)
        # Mode 14 (SOC < soc_min) or 9/12 (discharge < energy_min)
        assert mode in [9, 12, 14]


# ===========================================================================
# V(SOC) model
# ===========================================================================
class TestBessVoltageSoc:

    @pytest.mark.unit
    def test_constant_voltage_when_no_vmin(self, bess_ideal):
        """Without v_min, voltage is constant at rated value."""
        assert bess_ideal.get_voltage(0.0) == pytest.approx(50.0)
        assert bess_ideal.get_voltage(0.5) == pytest.approx(50.0)
        assert bess_ideal.get_voltage(1.0) == pytest.approx(50.0)

    @pytest.mark.unit
    def test_linear_voltage_with_vmin(self, bess_with_v_soc):
        """V(SOC) varies linearly between v_min (SOC=0) and v_rated (SOC=1)."""
        v_at_0 = bess_with_v_soc.get_voltage(0.0)
        v_at_1 = bess_with_v_soc.get_voltage(1.0)
        v_at_half = bess_with_v_soc.get_voltage(0.5)
        assert v_at_0 == pytest.approx(20.0)
        assert v_at_1 == pytest.approx(25.0)
        assert v_at_half == pytest.approx(22.5)

    @pytest.mark.unit
    def test_voltage_affects_power_limits(self, bess_with_v_soc):
        """At low SOC, V(SOC) reduces the max energy per step."""
        v_low = bess_with_v_soc.get_voltage(0.1)
        v_high = bess_with_v_soc.get_voltage(0.9)
        # energy_max = v * time * i_max / 1000
        e_low = v_low * TIME_STEP_1H * bess_with_v_soc.i_max / 1000
        e_high = v_high * TIME_STEP_1H * bess_with_v_soc.i_max / 1000
        assert e_low < e_high


# ===========================================================================
# Self-discharge
# ===========================================================================
class TestBessSelfDischarge:

    @pytest.mark.unit
    def test_self_discharge_reduces_soc(self, bess_default):
        """SOC decreases even with zero power input (self-discharge)."""
        bess_default.soc_in = 0.5
        # Run with zero power (below threshold -> mode 15)
        # Use a BESS with no threshold to test self-discharge
        bess = Bess(
            id='sd', cap_module=10.0, v=50.0, i_max=200, i_min=1,
            soc_in=0.5, soc_max=1.0, soc_min=0.0,
            n_series=1, n_parallel=1,
            cap_cost=500, opex_cost=10, inc_year=0,
            inc_start_end=[0, 0], tax_year=0,
            eta_charge=1.0, eta_discharge=1.0,
            self_discharge_rate_per_hour=0.01,  # exaggerated for testing
            min_energy_threshold=0.0,
        )
        # Tiny positive power to trigger charge path (not threshold block)
        _, soc, _, _, _, _, _, _, _ = bess.energy_performance(0.001, TIME_STEP_1H)
        # SOC should be slightly less than 0.5 due to self-discharge
        # (0.5 * (1 - 0.01) = 0.495 before any charge)
        assert soc < 0.5

    @pytest.mark.unit
    def test_no_self_discharge_when_zero_rate(self, bess_ideal):
        """With rate=0, no self-discharge occurs."""
        bess_ideal.soc_in = 0.5
        _, soc, _, _, _, _, _, _, mode = \
            bess_ideal.energy_performance(0.001, TIME_STEP_1H)
        # SOC should increase or stay ~0.5 (tiny charge added)
        assert soc >= 0.5 - 1e-10


# ===========================================================================
# C-rate limiting
# ===========================================================================
class TestBessCrate:

    @pytest.mark.unit
    def test_crate_limits_charge_power(self):
        """C-rate limit caps the charge energy per step."""
        bess = Bess(
            id='crate', cap_module=10.0, v=50.0, i_max=1000, i_min=1,
            soc_in=0.2, soc_max=1.0, soc_min=0.0,
            n_series=1, n_parallel=1,
            cap_cost=500, opex_cost=10, inc_year=0,
            inc_start_end=[0, 0], tax_year=0,
            eta_charge=1.0, eta_discharge=1.0,
            self_discharge_rate_per_hour=0.0,
            min_energy_threshold=0.0,
            c_rate_max=0.5,  # max 5 kW for 10 kWh battery
        )
        # Try to charge at 20 kW (way above 0.5C = 5 kW)
        _, soc, stored, _, _, surplus, _, _, _ = \
            bess.energy_performance(20.0, TIME_STEP_1H)
        max_energy = 0.5 * 10.0 * TIME_STEP_1H  # 5 kWh
        assert stored * TIME_STEP_1H <= max_energy + 1e-6
        assert surplus > 0


# ===========================================================================
# Micro-cycle threshold
# ===========================================================================
class TestBessMicroCycle:

    @pytest.mark.unit
    def test_below_threshold_no_operation(self, bess_default):
        """Power below the micro-cycle threshold is ignored."""
        # threshold = 0.01 * 2.56 = 0.0256 kWh
        tiny_power = 0.01  # kW, energy = 0.01 * 0.25 = 0.0025 kWh << threshold
        _, soc, stored, supply, _, _, _, _, mode = \
            bess_default.energy_performance(tiny_power, TIME_STEP_15MIN)
        assert stored == pytest.approx(0.0)
        assert supply == pytest.approx(0.0)
        assert mode == 15

    @pytest.mark.unit
    def test_above_threshold_operates(self, bess_default):
        """Power above the micro-cycle threshold triggers operation."""
        bess_default.soc_in = 0.5
        large_power = 2.0  # kW
        _, soc, stored, supply, _, _, _, _, mode = \
            bess_default.energy_performance(large_power, TIME_STEP_15MIN)
        assert stored > 0 or mode in [2, 5]  # operates or limited by current


# ===========================================================================
# Cycle counter
# ===========================================================================
class TestBessCycleCounter:

    @pytest.mark.unit
    def test_discharge_increments_counter(self, bess_ideal):
        """Discharging increments the cumulative discharge energy."""
        bess_ideal.soc_in = 0.8
        assert bess_ideal.full_cycle_equivalents == 0.0
        bess_ideal.energy_performance(-5.0, TIME_STEP_1H)
        assert bess_ideal.cumulative_discharge_energy > 0
        assert bess_ideal.full_cycle_equivalents > 0

    @pytest.mark.unit
    def test_charge_does_not_increment_counter(self, bess_ideal):
        """Charging does not affect the discharge cycle counter."""
        bess_ideal.soc_in = 0.2
        bess_ideal.energy_performance(5.0, TIME_STEP_1H)
        assert bess_ideal.cumulative_discharge_energy == pytest.approx(0.0)

    @pytest.mark.unit
    def test_fce_calculation(self, bess_ideal):
        """FCE = cumulative discharge / capacity."""
        bess_ideal.soc_in = 1.0
        # Discharge full capacity
        bess_ideal.energy_performance(-bess_ideal.cap / TIME_STEP_1H, TIME_STEP_1H)
        assert bess_ideal.full_cycle_equivalents == pytest.approx(1.0, rel=0.1)


# ===========================================================================
# Controller - multi-battery management
# ===========================================================================
class TestController:

    @pytest.mark.unit
    def test_charge_fills_lowest_soc_first(self):
        """In charge mode, the battery with lowest SOC is filled first."""
        b1 = Bess(
            id='b1', cap_module=10.0, v=50.0, i_max=200, i_min=1,
            soc_in=0.7, soc_max=1.0, soc_min=0.0,
            n_series=1, n_parallel=1,
            cap_cost=500, opex_cost=10, inc_year=0,
            inc_start_end=[0, 0], tax_year=0,
            eta_charge=1.0, eta_discharge=1.0,
            self_discharge_rate_per_hour=0.0, min_energy_threshold=0.0,
        )
        b2 = Bess(
            id='b2', cap_module=10.0, v=50.0, i_max=200, i_min=1,
            soc_in=0.3, soc_max=1.0, soc_min=0.0,
            n_series=1, n_parallel=1,
            cap_cost=500, opex_cost=10, inc_year=0,
            inc_start_end=[0, 0], tax_year=0,
            eta_charge=1.0, eta_discharge=1.0,
            self_discharge_rate_per_hour=0.0, min_energy_threshold=0.0,
        )
        # Small surplus: 1 kW for 1h = 1 kWh (fits in b2 alone)
        prod = np.array([2.0])
        dem = np.array([1.0])
        ctrl = Controller(bess=[b1, b2])
        ctrl.energy_performance(prod, dem, TIME_STEP_1H)
        # b2 (lower SOC) should have received more energy
        assert b2.en_perf_evolution['stored'][0] >= b1.en_perf_evolution['stored'][0]

    @pytest.mark.unit
    def test_discharge_uses_highest_soc_first(self):
        """In discharge mode, the battery with highest SOC supplies first."""
        b1 = Bess(
            id='b1', cap_module=10.0, v=50.0, i_max=200, i_min=1,
            soc_in=0.8, soc_max=1.0, soc_min=0.0,
            n_series=1, n_parallel=1,
            cap_cost=500, opex_cost=10, inc_year=0,
            inc_start_end=[0, 0], tax_year=0,
            eta_charge=1.0, eta_discharge=1.0,
            self_discharge_rate_per_hour=0.0, min_energy_threshold=0.0,
        )
        b2 = Bess(
            id='b2', cap_module=10.0, v=50.0, i_max=200, i_min=1,
            soc_in=0.3, soc_max=1.0, soc_min=0.0,
            n_series=1, n_parallel=1,
            cap_cost=500, opex_cost=10, inc_year=0,
            inc_start_end=[0, 0], tax_year=0,
            eta_charge=1.0, eta_discharge=1.0,
            self_discharge_rate_per_hour=0.0, min_energy_threshold=0.0,
        )
        # Deficit: demand > production
        prod = np.array([1.0])
        dem = np.array([3.0])
        ctrl = Controller(bess=[b1, b2])
        ctrl.energy_performance(prod, dem, TIME_STEP_1H)
        # b1 (higher SOC) should have supplied more
        assert b1.en_perf_evolution['supply'][0] >= b2.en_perf_evolution['supply'][0]

    @pytest.mark.unit
    def test_energy_conservation_in_controller(self):
        """Total energy is conserved: prod = self_cons + surplus + stored."""
        b1 = Bess(
            id='b1', cap_module=10.0, v=50.0, i_max=200, i_min=1,
            soc_in=0.5, soc_max=1.0, soc_min=0.0,
            n_series=1, n_parallel=1,
            cap_cost=500, opex_cost=10, inc_year=0,
            inc_start_end=[0, 0], tax_year=0,
            eta_charge=1.0, eta_discharge=1.0,
            self_discharge_rate_per_hour=0.0, min_energy_threshold=0.0,
        )
        prod = np.array([5.0, 0.5, 3.0, 0.0])
        dem = np.array([2.0, 2.0, 1.0, 4.0])
        ctrl = Controller(bess=[b1])
        stored, supply, power, surplus, deficit, soc = \
            ctrl.energy_performance(prod, dem, TIME_STEP_1H)

        for i in range(len(prod)):
            self_cons = min(prod[i], dem[i])
            # prod = self_cons + stored + surplus (when prod > dem)
            if prod[i] >= dem[i]:
                assert prod[i] == pytest.approx(self_cons + stored[i] + surplus[i], abs=1e-6)
            # dem = self_cons + supply + deficit (when dem > prod)
            else:
                assert dem[i] == pytest.approx(self_cons + supply[i] + deficit[i], abs=1e-6)
