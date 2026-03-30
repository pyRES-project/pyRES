"""
Tests for input validation across all classes.

Verifies that invalid parameters raise ValueError with clear messages
instead of producing silently incorrect results.
"""

import pytest
from src.rec_sim.System import System
from src.rec_sim.PvPanels import PvPanels
from src.rec_sim.Bess import Bess


# ===========================================================================
# System validation
# ===========================================================================
class TestSystemValidation:

    @pytest.mark.unit
    def test_negative_capacity(self):
        with pytest.raises(ValueError, match="capacity must be >= 0"):
            System(id='s', carriers=['e'], cap=-10, cap_cost=100, opex=1,
                   opex_cost=10, inc_year=0, inc_start_end=[0, 0], tax_year=0)

    @pytest.mark.unit
    def test_negative_cap_cost(self):
        with pytest.raises(ValueError, match="cap_cost must be >= 0"):
            System(id='s', carriers=['e'], cap=10, cap_cost=-100, opex=1,
                   opex_cost=10, inc_year=0, inc_start_end=[0, 0], tax_year=0)

    @pytest.mark.unit
    def test_invalid_inc_start_end(self):
        with pytest.raises(ValueError, match="inc_start_end"):
            System(id='s', carriers=['e'], cap=10, cap_cost=100, opex=1,
                   opex_cost=10, inc_year=0, inc_start_end=[5, 2], tax_year=0)

    @pytest.mark.unit
    def test_valid_system_passes(self):
        s = System(id='s', carriers=['e'], cap=10, cap_cost=100, opex=1,
                   opex_cost=10, inc_year=0, inc_start_end=[1, 5], tax_year=0)
        assert s.cap == 10


# ===========================================================================
# PvPanels validation
# ===========================================================================
class TestPvValidation:

    def _make_pv(self, **overrides):
        defaults = dict(
            id='pv', cap_cost=1500, opex_cost=40,
            inc_year=0, inc_start_end=[0, 0], tax_year=0,
            n_series=1, n_parallel=1,
        )
        defaults.update(overrides)
        return PvPanels(**defaults)

    @pytest.mark.unit
    def test_n_series_zero(self):
        with pytest.raises(ValueError, match="n_series"):
            self._make_pv(n_series=0)

    @pytest.mark.unit
    def test_n_parallel_negative(self):
        with pytest.raises(ValueError, match="n_parallel"):
            self._make_pv(n_parallel=-1)

    @pytest.mark.unit
    def test_imppt_ge_isc(self):
        with pytest.raises(ValueError, match="imppt_ref.*must be < isc_ref"):
            self._make_pv(imppt_ref=11.0, isc_ref=10.0)

    @pytest.mark.unit
    def test_vmppt_ge_voc(self):
        with pytest.raises(ValueError, match="vmppt_ref.*must be < voc_ref"):
            self._make_pv(vmppt_ref=50.0, voc_ref=49.0)

    @pytest.mark.unit
    def test_area_zero(self):
        with pytest.raises(ValueError, match="area must be > 0"):
            self._make_pv(area=0)

    @pytest.mark.unit
    def test_dc_ac_efficiency_over_one(self):
        with pytest.raises(ValueError, match="dc_ac_efficiency"):
            self._make_pv(dc_ac_efficiency=1.5)

    @pytest.mark.unit
    def test_dc_ac_efficiency_zero(self):
        with pytest.raises(ValueError, match="dc_ac_efficiency"):
            self._make_pv(dc_ac_efficiency=0)

    @pytest.mark.unit
    def test_negative_degradation(self):
        with pytest.raises(ValueError, match="annual_degradation"):
            self._make_pv(annual_degradation=-0.01)

    @pytest.mark.unit
    def test_bandgap_zero(self):
        with pytest.raises(ValueError, match="bandgap"):
            self._make_pv(eg=0)

    @pytest.mark.unit
    def test_valid_pv_passes(self):
        pv = self._make_pv()
        assert pv.cap > 0


# ===========================================================================
# Bess validation
# ===========================================================================
class TestBessValidation:

    def _make_bess(self, **overrides):
        defaults = dict(
            id='b', cap_module=2.56, v=25.6, i_max=100, i_min=5,
            soc_in=0.5, soc_max=0.8, soc_min=0.2,
            n_series=1, n_parallel=1,
            cap_cost=720, opex_cost=20,
            inc_year=0, inc_start_end=[0, 0], tax_year=0,
        )
        defaults.update(overrides)
        return Bess(**defaults)

    @pytest.mark.unit
    def test_soc_in_above_max(self):
        with pytest.raises(ValueError, match="soc_in.*must be in"):
            self._make_bess(soc_in=0.9, soc_max=0.8)

    @pytest.mark.unit
    def test_soc_in_below_min(self):
        with pytest.raises(ValueError, match="soc_in.*must be in"):
            self._make_bess(soc_in=0.1, soc_min=0.2)

    @pytest.mark.unit
    def test_soc_min_ge_max(self):
        with pytest.raises(ValueError, match="soc_min.*must be < soc_max"):
            self._make_bess(soc_min=0.8, soc_max=0.8, soc_in=0.8)

    @pytest.mark.unit
    def test_eta_charge_zero(self):
        with pytest.raises(ValueError, match="eta_charge"):
            self._make_bess(eta_charge=0)

    @pytest.mark.unit
    def test_eta_discharge_over_one(self):
        with pytest.raises(ValueError, match="eta_discharge"):
            self._make_bess(eta_discharge=1.1)

    @pytest.mark.unit
    def test_n_series_zero(self):
        with pytest.raises(ValueError, match="n_series"):
            self._make_bess(n_series=0)

    @pytest.mark.unit
    def test_negative_voltage(self):
        with pytest.raises(ValueError, match="rated voltage"):
            self._make_bess(v=-10)

    @pytest.mark.unit
    def test_negative_i_max(self):
        with pytest.raises(ValueError, match="i_max"):
            self._make_bess(i_max=-1)

    @pytest.mark.unit
    def test_negative_lifetime(self):
        with pytest.raises(ValueError, match="lifetime_years"):
            self._make_bess(lifetime_years=0)

    @pytest.mark.unit
    def test_capacity_fade_out_of_range(self):
        with pytest.raises(ValueError, match="annual_capacity_fade"):
            self._make_bess(annual_capacity_fade=1.0)

    @pytest.mark.unit
    def test_negative_crate(self):
        with pytest.raises(ValueError, match="c_rate_max"):
            self._make_bess(c_rate_max=-0.5)

    @pytest.mark.unit
    def test_valid_bess_passes(self):
        bess = self._make_bess()
        assert bess.cap > 0
