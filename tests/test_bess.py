"""
Unit tests for Bess and Controller classes.

Tests cover:
- Construction and parameter validation
- Charge/discharge with and without efficiency losses
- SOC limits (min, max)
- Self-discharge over time
- C-rate limiting
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
        """Total capacity = cap."""
        assert bess_default.cap == pytest.approx(2.56)

    @pytest.mark.unit
    def test_capacity_direct(self):
        """Capacity is directly specified."""
        bess = Bess(
            id='multi', cap=12.0, c_rate=1.0,
            soc_in=0.5, soc_max=1.0, soc_min=0.0,
            cap_cost=500, opex_cost=10, inc_year=0,
            inc_start_end=[0, 0], tax_year=0,
        )
        assert bess.cap == pytest.approx(12.0)

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
        _, soc, power_from_source, supply, power, surplus, deficit = \
            bess_ideal.energy_performance(power_in, TIME_STEP_1H)
        assert soc > soc_before
        assert power_from_source > 0
        assert supply == 0
        assert deficit == 0

    @pytest.mark.unit
    def test_charge_efficiency_loss(self, bess_default):
        """With eta_charge < 1, less energy is power_from_source than consumed from source."""
        bess_default.soc_in = 0.3
        power_in = 1.0  # kW
        _, soc, power_from_source, supply, power, surplus, deficit = \
            bess_default.energy_performance(power_in, TIME_STEP_1H)
        energy_in = power_in * TIME_STEP_1H
        energy_power_from_source_in_battery = bess_default.cap * soc - bess_default.cap * 0.3
        # The battery should store less than what was drawn (due to eta_charge)
        # Allow for self-discharge adjustment on initial SOC
        assert energy_power_from_source_in_battery < energy_in or power_from_source == pytest.approx(0.0)

    @pytest.mark.unit
    def test_charge_stops_at_soc_max(self, bess_ideal):
        """Battery does not charge beyond soc_max."""
        bess_ideal.soc_in = 0.95
        power_in = 100.0  # very large surplus
        _, soc, power_from_source, supply, power, surplus, deficit = \
            bess_ideal.energy_performance(power_in, TIME_STEP_1H)
        assert soc <= bess_ideal.soc_max + 1e-10
        assert surplus > 0  # excess not absorbed

    @pytest.mark.unit
    def test_charge_already_full(self, bess_ideal):
        """No charge when SOC already at max."""
        bess_ideal.soc_in = 1.0
        _, soc, power_from_source, supply, power, surplus, deficit = \
            bess_ideal.energy_performance(5.0, TIME_STEP_1H)
        assert power_from_source == pytest.approx(0.0)
        assert surplus == pytest.approx(5.0)


# ===========================================================================
# Discharge behavior
# ===========================================================================
class TestBessDischarge:

    @pytest.mark.unit
    def test_discharge_ideal_decreases_soc(self, bess_ideal):
        """Discharging with ideal battery decreases SOC."""
        soc_before = bess_ideal.soc_in
        power_in = -3.0  # kW deficit
        _, soc, power_from_source, supply, power, surplus, deficit = \
            bess_ideal.energy_performance(power_in, TIME_STEP_1H)
        assert soc < soc_before
        assert supply > 0
        assert power_from_source == 0

    @pytest.mark.unit
    def test_discharge_efficiency_loss(self):
        """With eta_discharge < 1, less energy is delivered than extracted from battery."""
        bess = Bess(
            id='eff', cap=10.0, c_rate=1.0,
            soc_in=0.8, soc_max=1.0, soc_min=0.0,
            cap_cost=500, opex_cost=10, inc_year=0,
            inc_start_end=[0, 0], tax_year=0,
            eta_charge=1.0, eta_discharge=0.90,
            self_discharge_rate_per_hour=0.0,
        )
        power_in = -2.0  # kW deficit
        _, soc, power_from_source, supply, power, surplus, deficit = \
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
        _, soc, power_from_source, supply, power, surplus, deficit = \
            bess_ideal.energy_performance(-100.0, TIME_STEP_1H)
        assert soc >= bess_ideal.soc_min - 1e-10
        assert deficit > 0

    @pytest.mark.unit
    def test_discharge_already_empty(self, bess_ideal):
        """No discharge when SOC at minimum."""
        bess_ideal.soc_in = 0.0
        _, soc, power_from_source, supply, power, surplus, deficit = \
            bess_ideal.energy_performance(-5.0, TIME_STEP_1H)
        assert supply == pytest.approx(0.0)
        assert deficit == pytest.approx(5.0)


# ===========================================================================
# Self-discharge
# ===========================================================================
class TestBessSelfDischarge:

    @pytest.mark.unit
    def test_self_discharge_reduces_soc(self, bess_default):
        """SOC decreases even with zero power input (self-discharge)."""
        bess_default.soc_in = 0.5
        # Use a BESS with exaggerated self-discharge for testing
        bess = Bess(
            id='sd', cap=10.0, c_rate=1.0,
            soc_in=0.5, soc_max=1.0, soc_min=0.0,
            cap_cost=500, opex_cost=10, inc_year=0,
            inc_start_end=[0, 0], tax_year=0,
            eta_charge=1.0, eta_discharge=1.0,
            self_discharge_rate_per_hour=0.01,  # exaggerated for testing
        )
        # Tiny positive power to trigger charge path
        _, soc, _, _, _, _, _ = bess.energy_performance(0.001, TIME_STEP_1H)
        # SOC should be slightly less than 0.5 due to self-discharge
        # (0.5 * (1 - 0.01) = 0.495 before any charge)
        assert soc < 0.5

    @pytest.mark.unit
    def test_no_self_discharge_when_zero_rate(self, bess_ideal):
        """With rate=0, no self-discharge occurs."""
        bess_ideal.soc_in = 0.5
        _, soc, _, _, _, _, _ = \
            bess_ideal.energy_performance(0.001, TIME_STEP_1H)
        # SOC should increase or stay ~0.5 (tiny charge added)
        assert soc >= 0.5 - 1e-10


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
            id='b1', cap=10.0, c_rate=1.0,
            soc_in=0.7, soc_max=1.0, soc_min=0.0,
            cap_cost=500, opex_cost=10, inc_year=0,
            inc_start_end=[0, 0], tax_year=0,
            eta_charge=1.0, eta_discharge=1.0,
            self_discharge_rate_per_hour=0.0,
        )
        b2 = Bess(
            id='b2', cap=10.0, c_rate=1.0,
            soc_in=0.3, soc_max=1.0, soc_min=0.0,
            cap_cost=500, opex_cost=10, inc_year=0,
            inc_start_end=[0, 0], tax_year=0,
            eta_charge=1.0, eta_discharge=1.0,
            self_discharge_rate_per_hour=0.0,
        )
        # Small surplus: 1 kW for 1h = 1 kWh (fits in b2 alone)
        prod = np.array([2.0])
        dem = np.array([1.0])
        ctrl = Controller(bess=[b1, b2])
        ctrl.energy_performance(prod, dem, TIME_STEP_1H)
        # b2 (lower SOC) should have received more energy
        assert b2.en_perf_evolution['power_from_source'][0] >= b1.en_perf_evolution['power_from_source'][0]

    @pytest.mark.unit
    def test_discharge_uses_highest_soc_first(self):
        """In discharge mode, the battery with highest SOC supplies first."""
        b1 = Bess(
            id='b1', cap=10.0, c_rate=1.0,
            soc_in=0.8, soc_max=1.0, soc_min=0.0,
            cap_cost=500, opex_cost=10, inc_year=0,
            inc_start_end=[0, 0], tax_year=0,
            eta_charge=1.0, eta_discharge=1.0,
            self_discharge_rate_per_hour=0.0,
        )
        b2 = Bess(
            id='b2', cap=10.0, c_rate=1.0,
            soc_in=0.3, soc_max=1.0, soc_min=0.0,
            cap_cost=500, opex_cost=10, inc_year=0,
            inc_start_end=[0, 0], tax_year=0,
            eta_charge=1.0, eta_discharge=1.0,
            self_discharge_rate_per_hour=0.0,
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
        """Total energy is conserved: prod = self_cons + surplus + power_from_source."""
        b1 = Bess(
            id='b1', cap=10.0, c_rate=1.0,
            soc_in=0.5, soc_max=1.0, soc_min=0.0,
            cap_cost=500, opex_cost=10, inc_year=0,
            inc_start_end=[0, 0], tax_year=0,
            eta_charge=1.0, eta_discharge=1.0,
            self_discharge_rate_per_hour=0.0,
        )
        prod = np.array([5.0, 0.5, 3.0, 0.0])
        dem = np.array([2.0, 2.0, 1.0, 4.0])
        ctrl = Controller(bess=[b1])
        power_from_source, supply, power, surplus, deficit, soc = \
            ctrl.energy_performance(prod, dem, TIME_STEP_1H)

        for i in range(len(prod)):
            self_cons = min(prod[i], dem[i])
            # prod = self_cons + power_from_source + surplus (when prod > dem)
            if prod[i] >= dem[i]:
                assert prod[i] == pytest.approx(self_cons + power_from_source[i] + surplus[i], abs=1e-6)
            # dem = self_cons + supply + deficit (when dem > prod)
            else:
                assert dem[i] == pytest.approx(self_cons + supply[i] + deficit[i], abs=1e-6)
