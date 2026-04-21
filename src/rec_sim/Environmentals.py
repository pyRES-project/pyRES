"""
Created on April 10 2026

@author: claude (GD)
"""

import numpy as np


class Environmentals:

    # ── Grid emission factor ──
    # Source: ISPRA / UNFCCC - Italian national inventory (2023 data)
    CO2_GRID_EMISSION_FACTOR = 256  # gCO2/kWh

    # ── Italian electricity grid fuel mix (Source: Terna/ISPRA 2023) ──
    GRID_FUEL_MIX = {
        'natural_gas': {'share': 0.49, 'fuel_rate': 0.187, 'unit': 'm³',
                        'label': 'Natural gas'},
        'oil':         {'share': 0.04, 'fuel_rate': 0.220, 'unit': 'kg',
                        'label': 'Oil / petroleum products'},
        'coal':        {'share': 0.04, 'fuel_rate': 0.325, 'unit': 'kg',
                        'label': 'Coal'},
    }

    # ── GWP embodied emission factors (cradle-to-gate) ──
    # PV: Frischknecht R. et al., IEA PVPS Task 12, Report T12-19:2020
    # BESS: Dai Q. et al., Argonne National Lab, ANL/ESD-19/2 (2019)
    # Inverter: Stolz P. et al., IEA PVPS Task 12 (2018)
    GWP_PV_KG_CO2_PER_KWP = 1150       # kg CO2-eq / kWp (module)
    GWP_BOS_KG_CO2_PER_KWP = 150       # kg CO2-eq / kWp (balance of system)
    GWP_INVERTER_KG_CO2_PER_KW = 50    # kg CO2-eq / kW (inverter)
    GWP_BESS_KG_CO2_PER_KWH = 120     # kg CO2-eq / kWh (cell + pack)

    # ── CRM intensity factors (LCA cradle-to-gate) ──
    # PV sources: IEA PVPS Task 12 (2020); ITRPV (2023); Fthenakis et al. (2012)
    # BESS sources: Dai et al., ANL (2019); EU JRC (2020); IRENA (2023)
    CRM_PV_PER_KWP = {
        'Silicon (met.)':    {'value': 3.5,  'unit': 'kg',
                              'source': 'IEA PVPS Task 12 (2020)'},
        'Silver':            {'value': 16.0, 'unit': 'g',
                              'source': 'ITRPV Roadmap (2023)'},
        'Copper':            {'value': 4.5,  'unit': 'kg',
                              'source': 'IEA PVPS Task 12 (2020)'},
        'Aluminium':         {'value': 20.0, 'unit': 'kg',
                              'source': 'IEA PVPS Task 12 (2020)'},
        'Lead (solder)':     {'value': 0.45, 'unit': 'kg',
                              'source': 'Fthenakis et al. (2012)'},
        'Tin (solder)':      {'value': 0.12, 'unit': 'kg',
                              'source': 'Fthenakis et al. (2012)'},
    }
    CRM_BESS_PER_KWH = {
        'Lithium (Li\u2082CO\u2083 eq.)': {'value': 0.60, 'unit': 'kg',
                                            'source': 'Dai et al., ANL (2019)'},
        'Cobalt':            {'value': 0.20, 'unit': 'kg',
                              'source': 'Dai et al., ANL (2019)'},
        'Nickel':            {'value': 0.40, 'unit': 'kg',
                              'source': 'Dai et al., ANL (2019)'},
        'Manganese':         {'value': 0.15, 'unit': 'kg',
                              'source': 'EU JRC CRM Report (2020)'},
        'Graphite (anode)':  {'value': 0.80, 'unit': 'kg',
                              'source': 'Dai et al., ANL (2019)'},
        'Copper':            {'value': 0.80, 'unit': 'kg',
                              'source': 'IRENA (2023)'},
    }

    def __init__(self, components, annual_prod_kwh, time_horizon):
        """
        Environmentals computes environmental performance indicators for a prosumer or REC,
        including CO2 emissions avoided, fossil fuel savings, GWP lifecycle assessment
        (cradle-to-gate embodied emissions of PV, balance of system, inverter and BESS,
        including replacement cycles) and a Critical Raw Materials (CRM) inventory.

        The class follows the same pattern as Economics: it receives a list of components
        (systems + BESS) and annual energy flows, then returns a dictionary of environmental KPIs.

        References:
        [1] ISPRA / UNFCCC - Italian national inventory (2023 data) for grid emission factor.
        [2] Terna / ISPRA (2023) for Italian electricity grid fuel mix.
        [3] Frischknecht R. et al., IEA PVPS Task 12, Report T12-19:2020 for PV GWP.
        [4] Dai Q. et al., Argonne National Lab, ANL/ESD-19/2 (2019) for BESS GWP and CRM.
        [5] Stolz P. et al., IEA PVPS Task 12 (2018) for inverter GWP.
        [6] ITRPV Roadmap (2023), Fthenakis et al. (2012), EU JRC CRM Report (2020), IRENA (2023) for CRM intensities.

        :param components: list --> list of System/Bess objects owned by the prosumer or REC. Each must expose cap and lifetime_years; BESS objects are detected by the presence of soc_max/soc_min attributes.
        :param annual_prod_kwh: float --> (kWh/year) total annual energy production from the components (year 1 baseline).
        :param time_horizon: int --> (year) investment time horizon over which avoided emissions and component replacements are computed.
        """
        self.components = components
        self.annual_prod_kwh = annual_prod_kwh
        self.time_horizon = time_horizon

    def compute_environmental(self):
        """
        Compute environmental performance indicators for the components provided at init.

        :return: env_perf: dict --> dictionary of environmental KPIs with the following keys:
            - co2_grid_factor:         float --> (gCO2/kWh) grid emission factor assumed for avoided emissions.
            - co2_avoided_annual_t:    float --> (tCO2/year) avoided emissions in year 1.
            - co2_avoided_lifetime_t:  float --> (tCO2) cumulative avoided emissions over time_horizon.
            - fuel_savings:            dict  --> per-fuel annual and lifetime savings, keyed by fuel (natural_gas, oil, coal). Each entry has label, annual, lifetime and unit.
            - gwp_pv_modules_t:        float --> (tCO2-eq) embodied GWP of PV modules.
            - gwp_bos_t:               float --> (tCO2-eq) embodied GWP of balance of system.
            - gwp_inverter_t:          float --> (tCO2-eq) embodied GWP of inverter (initial + replacements every 10 years).
            - gwp_bess_t:              float --> (tCO2-eq) embodied GWP of BESS (initial + replacements based on lifetime_years).
            - gwp_embodied_t:          float --> (tCO2-eq) total embodied emissions (PV + BOS + inverter + BESS + replacements).
            - gwp_net_t:               float --> (tCO2-eq) net lifecycle balance = embodied - avoided (negative = net environmental benefit).
            - co2_payback_years:       float --> (year) time required for avoided emissions to offset embodied emissions.
            - lifecycle_ef:            float --> (gCO2-eq/kWh) lifecycle emission factor of the produced energy.
            - total_pv_kwp:            float --> (kWp) total installed PV capacity across components.
            - total_bess_kwh:          float --> (kWh) total installed BESS capacity across components.
            - crm_pv:                  dict  --> per-material CRM inventory for PV components (intensity, unit, total_raw, total_kg, source).
            - crm_pv_total_kg:         float --> (kg) total critical raw materials in PV components.
            - crm_bess:                dict  --> per-material CRM inventory for BESS components (same sub-keys as crm_pv).
            - crm_bess_total_kg:       float --> (kg) total critical raw materials in BESS components.
            - crm_total_kg:            float --> (kg) overall total CRM (PV + BESS).
        """

        # ── Classify components ──
        systems = [c for c in self.components if not self._is_bess(c)]
        bess_list = [c for c in self.components if self._is_bess(c)]

        total_pv_kwp = sum(s.cap for s in systems)
        total_bess_kwh = sum(b.cap for b in bess_list)

        # ── CO2 avoided ──
        co2_avoided_annual_t = (self.annual_prod_kwh
                                * self.CO2_GRID_EMISSION_FACTOR / 1e6)
        co2_avoided_lifetime_t = co2_avoided_annual_t * self.time_horizon

        # ── Fossil fuel savings ──
        fuel_savings = {}
        for fuel_key, fuel_data in self.GRID_FUEL_MIX.items():
            displaced_kwh = self.annual_prod_kwh * fuel_data['share']
            annual_saving = displaced_kwh * fuel_data['fuel_rate']
            fuel_savings[fuel_key] = {
                'label': fuel_data['label'],
                'annual': annual_saving,
                'lifetime': annual_saving * self.time_horizon,
                'unit': fuel_data['unit'],
            }

        # ── GWP lifecycle ──
        gwp_pv_modules = self.GWP_PV_KG_CO2_PER_KWP * total_pv_kwp
        gwp_bos = self.GWP_BOS_KG_CO2_PER_KWP * total_pv_kwp
        gwp_inverter = self.GWP_INVERTER_KG_CO2_PER_KW * total_pv_kwp
        gwp_bess_init = self.GWP_BESS_KG_CO2_PER_KWH * total_bess_kwh

        # Inverter replacement (once every 10 years)
        n_inv_repl = max(0, self.time_horizon // 10 - 1)
        gwp_inverter_repl = gwp_inverter * n_inv_repl

        # Component replacement (based on lifetime_years attribute)
        gwp_repl_components = 0
        for c in self.components:
            lifetime = c.lifetime_years
            if lifetime is not None and lifetime > 0:
                n_repl = max(0, self.time_horizon // lifetime - 1)
                if self._is_bess(c):
                    gwp_repl_components += (self.GWP_BESS_KG_CO2_PER_KWH
                                           * c.cap * n_repl)
                else:
                    gwp_repl_components += (self.GWP_PV_KG_CO2_PER_KWP
                                           * c.cap * n_repl)

        gwp_embodied = (gwp_pv_modules + gwp_bos + gwp_inverter
                        + gwp_bess_init + gwp_inverter_repl + gwp_repl_components)
        gwp_embodied_t = gwp_embodied / 1000

        gwp_net_t = gwp_embodied_t - co2_avoided_lifetime_t

        co2_payback_years = (gwp_embodied_t / co2_avoided_annual_t
                             if co2_avoided_annual_t > 0 else float('inf'))

        total_lifetime_prod_kwh = self.annual_prod_kwh * self.time_horizon
        lifecycle_ef = ((gwp_embodied * 1000) / total_lifetime_prod_kwh
                        if total_lifetime_prod_kwh > 0 else 0)

        # ── CRM inventory ──
        crm_pv = {}
        crm_pv_total_kg = 0
        for mat, info in self.CRM_PV_PER_KWP.items():
            raw = info['value'] * total_pv_kwp
            kg = raw / 1000 if info['unit'] == 'g' else raw
            crm_pv[mat] = {
                'intensity': info['value'],
                'unit': info['unit'],
                'total_raw': raw,
                'total_kg': kg,
                'source': info['source'],
            }
            crm_pv_total_kg += kg

        crm_bess = {}
        crm_bess_total_kg = 0
        for mat, info in self.CRM_BESS_PER_KWH.items():
            raw = info['value'] * total_bess_kwh
            kg = raw / 1000 if info['unit'] == 'g' else raw
            crm_bess[mat] = {
                'intensity': info['value'],
                'unit': info['unit'],
                'total_raw': raw,
                'total_kg': kg,
                'source': info['source'],
            }
            crm_bess_total_kg += kg

        crm_total_kg = crm_pv_total_kg + crm_bess_total_kg

        # ── Build result dict ──
        env_perf = {
            'co2_grid_factor': self.CO2_GRID_EMISSION_FACTOR,
            'co2_avoided_annual_t': co2_avoided_annual_t,
            'co2_avoided_lifetime_t': co2_avoided_lifetime_t,
            'fuel_savings': fuel_savings,
            'gwp_pv_modules_t': gwp_pv_modules / 1000,
            'gwp_bos_t': gwp_bos / 1000,
            'gwp_inverter_t': (gwp_inverter + gwp_inverter_repl) / 1000,
            'gwp_bess_t': (gwp_bess_init + gwp_repl_components) / 1000,
            'gwp_embodied_t': gwp_embodied_t,
            'gwp_net_t': gwp_net_t,
            'co2_payback_years': co2_payback_years,
            'lifecycle_ef': lifecycle_ef,
            'total_pv_kwp': total_pv_kwp,
            'total_bess_kwh': total_bess_kwh,
            'crm_pv': crm_pv,
            'crm_pv_total_kg': crm_pv_total_kg,
            'crm_bess': crm_bess,
            'crm_bess_total_kg': crm_bess_total_kg,
            'crm_total_kg': crm_total_kg,
        }

        return env_perf

    @staticmethod
    def _is_bess(component):
        """Distinguish BESS from System by checking for BESS-specific attributes."""
        return hasattr(component, 'soc_max') and hasattr(component, 'soc_min')