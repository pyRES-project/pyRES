"""
Unit tests for PvPanels class.

Tests cover:
- Construction, capacity and array scaling
- Series resistance calculation (bisection convergence)
- Irradiance computation with and without IAM
- Power output: zero at night, positive during day
- System derate factor application
- Fill factor and efficiency calculations
- Thermal model behavior
- Configurable bandgap
"""

import pytest
import numpy as np
from src.rec_sim.PvPanels import PvPanels
from tests.conftest import TIME_STEP_15MIN, N_STEPS_DAY


# ===========================================================================
# Construction and scaling
# ===========================================================================
class TestPvConstruction:

    @pytest.mark.unit
    def test_single_module_capacity(self, pv_default):
        """Capacity of a single module = Vmppt * Imppt / 1000."""
        expected_cap = 40.6 * 9.86 / 1000
        assert pv_default.cap_module == pytest.approx(expected_cap, rel=1e-4)
        assert pv_default.cap == pytest.approx(expected_cap, rel=1e-4)

    @pytest.mark.unit
    def test_array_capacity_scales(self, pv_array_3x2):
        """Array capacity = module capacity * n_series * n_parallel."""
        expected = pv_array_3x2.cap_module * 3 * 2
        assert pv_array_3x2.cap == pytest.approx(expected, rel=1e-4)

    @pytest.mark.unit
    def test_array_area(self, pv_array_3x2):
        """Array area = module area * n_series * n_parallel."""
        expected = 2.07 * 3 * 2
        assert pv_array_3x2.array_area == pytest.approx(expected)

    @pytest.mark.unit
    def test_reference_efficiency(self, pv_default):
        """Reference efficiency = (Imppt * Vmppt) / (I_tot_ref * area)."""
        expected = (9.86 * 40.6) / (1000 * 2.07)
        assert pv_default.eff_ref == pytest.approx(expected, rel=1e-4)


# ===========================================================================
# Series resistance (bisection)
# ===========================================================================
class TestRserie:

    @pytest.mark.unit
    def test_rserie_positive(self, pv_default):
        """Series resistance must be positive."""
        assert pv_default.r_serie > 0

    @pytest.mark.unit
    def test_rserie_reasonable_range(self, pv_default):
        """For a single module, Rs is typically 0.1 - 2.0 ohm."""
        # For the array, Rs is scaled by n_series/n_parallel
        # For a 1x1 module, check raw value
        assert 0.01 < pv_default.r_serie < 5.0


# ===========================================================================
# System derate factor
# ===========================================================================
class TestSystemDerate:

    @pytest.mark.unit
    def test_derate_no_losses(self, pv_no_losses):
        """With all losses zero, derate factor = 1.0."""
        assert pv_no_losses.system_derate == pytest.approx(1.0)

    @pytest.mark.unit
    def test_derate_with_losses(self, pv_array_3x2):
        """Derate = eta_inv * (1-mismatch) * (1-wiring) * (1-soiling)."""
        expected = 0.97 * (1 - 0.02) * (1 - 0.015) * (1 - 0.03)
        assert pv_array_3x2.system_derate == pytest.approx(expected, rel=1e-6)

    @pytest.mark.unit
    def test_derate_less_than_one(self, pv_array_3x2):
        """Derate factor must be < 1 when losses are present."""
        assert pv_array_3x2.system_derate < 1.0

    @pytest.mark.unit
    def test_derate_applied_to_output(self, pv_no_losses, pv_array_3x2,
                                       synthetic_irradiance_day):
        """Output with losses < output without losses (same array config)."""
        irr = synthetic_irradiance_day
        # Build a pv_with_losses matching pv_no_losses geometry
        pv_with_losses = PvPanels(
            id='pv_loss', cap_cost=1500, opex_cost=40,
            inc_year=0, inc_start_end=[0, 0], tax_year=0,
            n_series=1, n_parallel=1,
            dc_ac_efficiency=0.95, mismatch_loss=0.05,
            wiring_loss=0.02, soiling_loss=0.04,
        )
        pv_no_losses.compute_output(
            slope=30, theta=None,
            I_beam=irr['I_beam'], I_skydiff=irr['I_skydiff'],
            I_grounddiff=irr['I_grounddiff'], t_amb=irr['t_amb'],
        )
        pv_with_losses.compute_output(
            slope=30, theta=None,
            I_beam=irr['I_beam'], I_skydiff=irr['I_skydiff'],
            I_grounddiff=irr['I_grounddiff'], t_amb=irr['t_amb'],
        )
        prod_no_loss = np.sum(pv_no_losses.en_perf_evolution['electricity']['prod'])
        prod_with_loss = np.sum(pv_with_losses.en_perf_evolution['electricity']['prod'])
        assert prod_with_loss < prod_no_loss


# ===========================================================================
# Irradiance computation
# ===========================================================================
class TestIrradiance:

    @pytest.mark.unit
    def test_total_radiation_no_iam(self, pv_default, synthetic_irradiance_day):
        """Without theta, I_total = beam + skydiff + grounddiff."""
        irr = synthetic_irradiance_day
        I_total = pv_default.compute_total_radiation(
            slope=30, theta=None,
            I_beam=irr['I_beam'], I_skydiff=irr['I_skydiff'],
            I_grounddiff=irr['I_grounddiff'],
        )
        expected = np.array(irr['I_beam']) + np.array(irr['I_skydiff']) + np.array(irr['I_grounddiff'])
        np.testing.assert_allclose(I_total, expected, atol=1e-6)

    @pytest.mark.unit
    def test_total_radiation_with_iam_at_high_angle(self, pv_default, synthetic_irradiance_day):
        """With IAM at high angle of incidence (60°), I_total is reduced vs no IAM."""
        irr = synthetic_irradiance_day
        # High incidence angle: IAM correction is more pronounced
        theta = np.full(irr['n_steps'], 60.0)
        I_total_iam = pv_default.compute_total_radiation(
            slope=30, theta=theta,
            I_beam=irr['I_beam'], I_skydiff=irr['I_skydiff'],
            I_grounddiff=irr['I_grounddiff'],
        )
        I_total_no_iam = pv_default.compute_total_radiation(
            slope=30, theta=None,
            I_beam=irr['I_beam'], I_skydiff=irr['I_skydiff'],
            I_grounddiff=irr['I_grounddiff'],
        )
        # At high angles, IAM significantly reduces beam component
        assert np.sum(I_total_iam) < np.sum(I_total_no_iam)

    @pytest.mark.unit
    def test_zero_irradiance_gives_zero_total(self, pv_default):
        """Zero irradiance produces zero total radiation."""
        zeros = np.zeros(10)
        I_total = pv_default.compute_total_radiation(
            slope=30, theta=None,
            I_beam=zeros, I_skydiff=zeros, I_grounddiff=zeros,
        )
        np.testing.assert_allclose(I_total, 0.0)


# ===========================================================================
# Power output
# ===========================================================================
class TestPowerOutput:

    @pytest.mark.unit
    def test_zero_irradiance_zero_power(self, pv_default):
        """No irradiance produces zero power output."""
        n = 10
        I_total, vmp, imp, p_max, voc, isc, t_cell, ff, eff = \
            pv_default.compute_output(
                slope=30, theta=None,
                I_beam=np.zeros(n), I_skydiff=np.zeros(n),
                I_grounddiff=np.zeros(n), t_amb=np.full(n, 25.0),
            )
        np.testing.assert_allclose(p_max, 0.0)
        assert np.all(pv_default.en_perf_evolution['electricity']['prod'] == 0)

    @pytest.mark.unit
    def test_positive_irradiance_positive_power(self, pv_default, synthetic_irradiance_day):
        """Positive irradiance during daytime produces positive power."""
        irr = synthetic_irradiance_day
        pv_default.compute_output(
            slope=30, theta=None,
            I_beam=irr['I_beam'], I_skydiff=irr['I_skydiff'],
            I_grounddiff=irr['I_grounddiff'], t_amb=irr['t_amb'],
        )
        prod = pv_default.en_perf_evolution['electricity']['prod']
        # Should have some positive production during daytime
        assert np.max(prod) > 0
        # No negative production ever
        assert np.all(prod >= 0)

    @pytest.mark.unit
    def test_production_in_kw(self, pv_default, synthetic_irradiance_day):
        """Production output is in kW (not W) and within reasonable range for 1 module."""
        irr = synthetic_irradiance_day
        pv_default.compute_output(
            slope=30, theta=None,
            I_beam=irr['I_beam'], I_skydiff=irr['I_skydiff'],
            I_grounddiff=irr['I_grounddiff'], t_amb=irr['t_amb'],
        )
        prod = pv_default.en_perf_evolution['electricity']['prod']
        # Single module ~400W peak = 0.4 kW
        assert np.max(prod) < 1.0  # < 1 kW for single module
        assert np.max(prod) > 0.05  # at least some output


# ===========================================================================
# Fill factor and efficiency
# ===========================================================================
class TestFillFactorEfficiency:

    @pytest.mark.unit
    def test_fill_factor_range(self, pv_default, synthetic_irradiance_day):
        """Fill factor should be in [0, 1] range."""
        irr = synthetic_irradiance_day
        I_total, vmp, imp, p_max, voc, isc, t_cell, ff, eff = \
            pv_default.compute_output(
                slope=30, theta=None,
                I_beam=irr['I_beam'], I_skydiff=irr['I_skydiff'],
                I_grounddiff=irr['I_grounddiff'], t_amb=irr['t_amb'],
            )
        assert np.all(ff >= 0)
        assert np.all(ff <= 1.0)

    @pytest.mark.unit
    def test_efficiency_range(self, pv_default, synthetic_irradiance_day):
        """Efficiency should be in [0, ~0.25] range for silicon."""
        irr = synthetic_irradiance_day
        I_total, vmp, imp, p_max, voc, isc, t_cell, ff, eff = \
            pv_default.compute_output(
                slope=30, theta=None,
                I_beam=irr['I_beam'], I_skydiff=irr['I_skydiff'],
                I_grounddiff=irr['I_grounddiff'], t_amb=irr['t_amb'],
            )
        assert np.all(eff >= 0)
        assert np.all(eff <= 0.30)

    @pytest.mark.unit
    def test_fill_factor_zero_at_night(self, pv_default, synthetic_irradiance_day):
        """Fill factor is zero when there is no irradiance (night)."""
        irr = synthetic_irradiance_day
        _, _, _, _, _, _, _, ff, _ = pv_default.compute_output(
            slope=30, theta=None,
            I_beam=irr['I_beam'], I_skydiff=irr['I_skydiff'],
            I_grounddiff=irr['I_grounddiff'], t_amb=irr['t_amb'],
        )
        # Night hours (0-5h, 19-24h) -> steps 0-19 and 76-95
        night_ff = np.concatenate([ff[:20], ff[76:]])
        np.testing.assert_allclose(night_ff, 0.0, atol=1e-10)


# ===========================================================================
# Thermal model
# ===========================================================================
class TestThermalModel:

    @pytest.mark.unit
    def test_cell_temp_above_ambient(self, pv_default, synthetic_irradiance_day):
        """Cell temperature should be above ambient when irradiance > 0."""
        irr = synthetic_irradiance_day
        _, _, _, _, _, _, t_cell, _, _ = pv_default.compute_output(
            slope=30, theta=None,
            I_beam=irr['I_beam'], I_skydiff=irr['I_skydiff'],
            I_grounddiff=irr['I_grounddiff'], t_amb=irr['t_amb'],
        )
        t_amb_k = np.array(irr['t_amb']) + 273.15
        daytime = (np.array(irr['I_beam']) + np.array(irr['I_skydiff'])) > 10
        # During daytime, cell temp > ambient
        assert np.all(t_cell[daytime] >= t_amb_k[daytime] - 0.1)

    @pytest.mark.unit
    def test_cell_temp_equals_ambient_at_night(self, pv_default, synthetic_irradiance_day):
        """Cell temperature equals ambient when irradiance < 1 W/m2."""
        irr = synthetic_irradiance_day
        _, _, _, _, _, _, t_cell, _, _ = pv_default.compute_output(
            slope=30, theta=None,
            I_beam=irr['I_beam'], I_skydiff=irr['I_skydiff'],
            I_grounddiff=irr['I_grounddiff'], t_amb=irr['t_amb'],
        )
        t_amb_k = np.array(irr['t_amb']) + 273.15
        nighttime = (np.array(irr['I_beam']) + np.array(irr['I_skydiff'])) < 1
        np.testing.assert_allclose(t_cell[nighttime], t_amb_k[nighttime], atol=0.5)

    @pytest.mark.unit
    def test_wind_reduces_cell_temp(self, pv_default, synthetic_irradiance_day):
        """Higher wind speed reduces cell temperature (Faiman model)."""
        irr = synthetic_irradiance_day
        # Low wind
        _, _, _, _, _, _, t_cell_low, _, _ = pv_default.compute_output(
            slope=30, theta=None,
            I_beam=irr['I_beam'], I_skydiff=irr['I_skydiff'],
            I_grounddiff=irr['I_grounddiff'], t_amb=irr['t_amb'],
            wind_speed=np.ones(irr['n_steps']) * 0.5,
        )
        # High wind
        _, _, _, _, _, _, t_cell_high, _, _ = pv_default.compute_output(
            slope=30, theta=None,
            I_beam=irr['I_beam'], I_skydiff=irr['I_skydiff'],
            I_grounddiff=irr['I_grounddiff'], t_amb=irr['t_amb'],
            wind_speed=np.ones(irr['n_steps']) * 10.0,
        )
        daytime = (np.array(irr['I_beam']) + np.array(irr['I_skydiff'])) > 50
        # Higher wind -> lower cell temperature
        assert np.mean(t_cell_high[daytime]) < np.mean(t_cell_low[daytime])


# ===========================================================================
# Bandgap configurability
# ===========================================================================
class TestBandgap:

    @pytest.mark.unit
    def test_default_bandgap_silicon(self, pv_default):
        """Default bandgap is 1.12 eV (silicon)."""
        assert pv_default.eg == pytest.approx(1.12)

    @pytest.mark.unit
    def test_custom_bandgap(self):
        """Custom bandgap is stored correctly."""
        pv = PvPanels(
            id='gaas', cap_cost=2000, opex_cost=50,
            inc_year=0, inc_start_end=[0, 0], tax_year=0,
            eg=1.35,
        )
        assert pv.eg == pytest.approx(1.35)
