"""
Multi-objective PV + BESS sizing optimizer for prosumers using NSGA-II.

Optimizes PV array size and BESS capacity for a single prosumer to find
Pareto-optimal trade-offs between energy self-use and economic return (NPV).

Two objective modes are available:
    - 'self_consumption': maximize absolute self-consumption [MWh/y] + NPV [€]
      Best for investment decisions on a specific site.
    - 'self_sufficiency': maximize self-sufficiency rate [-] + NPV [€]
      Best for parametric studies comparing multiple sites/prosumers.

Decision variables:
    X[0] = n_parallel_pv   (number of PV strings in parallel; n_series_pv is fixed)
    X[1] = n_parallel_bess (number of BESS units in parallel)

Constraints:
    g1: production / demand <= max_prod_dem_ratio
    g2: self-consumption ratio >= min_self_cons_ratio
    g3: CAPEX(PV+BESS) <= budget_max

Performance note:
    PV output is precomputed once for a single string (n_parallel=1) and then
    linearly scaled by X[0] in each evaluation, avoiding repeated PvPanels
    initialization and compute_output calls (~100x speedup).

Usage:
    from src.optimizer.prosumer_optimizer import ProsumerOptimizer
    from src.kernel.run import fetch_meteo

    meteo = fetch_meteo(lat=41.9, lon=12.5, tilt=30, azimuth=0,
                        time_step_str='15min', time_step_hours=0.25)
    I_beam, I_skydiff, I_grounddiff, t_amb, wind_speed, theta = meteo

    optimizer = ProsumerOptimizer(
        demand=demand_kw_array,
        meteo={'I_beam': I_beam, 'I_skydiff': I_skydiff,
               'I_grounddiff': I_grounddiff, 't_amb': t_amb,
               'wind_speed': wind_speed, 'theta': theta},
    )
    results_df = optimizer.optimize()
    optimizer.plot_pareto()

@author: giovanni
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.termination import get_termination
from pymoo.termination.default import DefaultMultiObjectiveTermination
from pymoo.termination.collection import TerminationCollection
from pymoo.optimize import minimize

from src.rec_sim.PvPanels import PvPanels
from src.rec_sim.Bess import Bess
from src.rec_sim.Controller import Controller
from src.rec_sim.Economics import Economics


# ---------------------------------------------------------------------------
# Default parameters (NeON 2 LG370Q1C-V5, 370 W, 72 cells)
# ---------------------------------------------------------------------------
DEFAULT_PV_MODULE = {
    'isc_ref': 10.47,
    'voc_ref': 49.3,
    'vmppt_ref': 40.6,
    'imppt_ref': 9.86,
    'mu_isc_ref': 0.02,
    'mu_voc_ref': 0.26,
    'ser_cell': 60,
    't_cell_noct_c': 42,
    'area': 2.07,
    'eg': 1.12,
    'dc_ac_efficiency': 0.97,
    'mismatch_loss': 0.02,
    'wiring_loss': 0.015,
    'soiling_loss': 0.03,
}

# Default BESS parameters (Li-ion, 2.56 kWh unit)
DEFAULT_BESS_TECH = {
    'cap_unit': 2.560,        # kWh per unit (used for sizing granularity)
    'c_rate': 1.0,
    'soc_in': 0.2,
    'soc_max': 0.8,
    'soc_min': 0.2,
    'eta_charge': 0.95,
    'eta_discharge': 0.95,
    'self_discharge_rate_per_hour': 0.00004,
    'lifetime_years': 15,
}

DEFAULT_PV_ECONOMICS = {
    'cap_cost': 1000,          # €/kW
    'opex_cost': 40,           # €/kW/year
    'inc_year': 0,             # €/year (flat incentive, alternative to inc_rate)
    'inc_start_end': [0, 0],
    'tax_year': 0,
    'inverter_cost': 350,      # €/kW replacement cost
    'inverter_year': 10,       # year of inverter replacement
    'inc_rate': 0.0,           # fractional incentive on CAPEX (e.g. 0.5 = 50%)
    'inc_cap': 96000,          # max total incentive €
    'inc_duration': 10,        # years over which incentive is spread
}

DEFAULT_BESS_ECONOMICS = {
    'cap_cost': 450,           # €/kWh
    'opex_cost': 20,           # €/kWh/year
    'inc_year': 0,
    'inc_start_end': [0, 0],
    'tax_year': 0,
    'inc_rate': 0.0,
    'inc_cap': 96000,
    'inc_duration': 10,
}

DEFAULT_PROSUMER_ECONOMICS = {
    'tax_rate': 0.2,
    'int_rate': 0.05,
    'price_buy': 250,          # €/MWh
    'price_sold': 104,         # €/MWh
    'decay': 0.02,
    'prod_degradation': 0.005,
}


# ---------------------------------------------------------------------------
# Lightweight proxies for Economics (avoid expensive PvPanels re-init)
# ---------------------------------------------------------------------------
class _SystemProxy:
    """Minimal object satisfying the Economics component interface."""

    def __init__(self, id, cap, cap_cost, opex, opex_cost,
                 inc_year, inc_start_end, tax_year, other_cost, other_rev):
        self.id = id
        self.cap = cap
        self.cap_cost_unit = cap_cost
        self.opex = opex
        self.opex_cost = opex_cost
        self.inc_year = inc_year
        self.inc_start_end = inc_start_end
        self.tax_year = tax_year
        self.other_cost = other_cost
        self.other_rev = other_rev


class _BessProxy(_SystemProxy):
    """Extends _SystemProxy with BESS replacement attributes."""

    def __init__(self, lifetime_years, **kwargs):
        super().__init__(**kwargs)
        self.lifetime_years = lifetime_years


# ---------------------------------------------------------------------------
# Main optimizer class
# ---------------------------------------------------------------------------
class ProsumerOptimizer:
    """Multi-objective PV + BESS sizing optimizer using NSGA-II.

    Parameters
    ----------
    demand : array-like
        Electrical load profile in kW (quarter-hourly or hourly).
    meteo : dict
        Keys: 'I_beam', 'I_skydiff', 'I_grounddiff', 't_amb'.
        Optional: 'wind_speed', 'theta'.
    time_step : float
        Time step in hours (0.25 for 15 min, 1 for hourly).
    pv_module_params : dict, optional
        Override DEFAULT_PV_MODULE entries.
    pv_economics : dict, optional
        Override DEFAULT_PV_ECONOMICS entries.
    n_series_pv : int
        Fixed number of PV modules in series per string.
    tilt : float
        Panel tilt angle in degrees.
    bess_tech_params : dict, optional
        Override DEFAULT_BESS_TECH entries.
    bess_economics : dict, optional
        Override DEFAULT_BESS_ECONOMICS entries.
    space_max : float
        Maximum available surface for PV installation [m²].
    utilization_factor : float
        Fraction of space_max usable for panels (0-1).
    specific_budget : float
        Maximum budget in €/kWp referred to the net installable peak power.
    min_self_cons_ratio : float
        Minimum self-consumption/demand ratio constraint (0-1).
    max_prod_dem_ratio : float
        Maximum production/demand ratio constraint.
    prosumer_economics : dict, optional
        Override DEFAULT_PROSUMER_ECONOMICS entries.
    time_horizon : int
        Investment analysis horizon in years.
    objective_mode : str
        'self_consumption' or 'self_sufficiency'.
    pop_size : int
        NSGA-II population size.
    n_gen : int
        Number of NSGA-II generations.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(self,
                 demand,
                 meteo,
                 time_step=0.25,
                 # PV
                 pv_module_params=None,
                 pv_economics=None,
                 n_series_pv=10,
                 tilt=30,
                 # BESS
                 bess_tech_params=None,
                 bess_economics=None,
                 # Constraints
                 space_max=100.0,
                 utilization_factor=0.6,
                 specific_budget=2000.0,
                 min_self_cons_ratio=0.3,
                 max_prod_dem_ratio=2.0,
                 # Prosumer economics
                 prosumer_economics=None,
                 time_horizon=20,
                 # NSGA-II
                 objective_mode='self_consumption',
                 pop_size=100,
                 n_gen=50,
                 seed=42):

        if objective_mode not in ('self_consumption', 'self_sufficiency'):
            raise ValueError(f"objective_mode must be 'self_consumption' or 'self_sufficiency', got '{objective_mode}'")

        self.time_step = time_step
        self.time_horizon = time_horizon
        self.objective_mode = objective_mode
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.seed = seed

        # Merge user overrides with defaults
        self.pv_module = {**DEFAULT_PV_MODULE, **(pv_module_params or {})}
        self.bess_tech = {**DEFAULT_BESS_TECH, **(bess_tech_params or {})}
        self.pv_econ = {**DEFAULT_PV_ECONOMICS, **(pv_economics or {})}
        self.bess_econ = {**DEFAULT_BESS_ECONOMICS, **(bess_economics or {})}
        self.pros_econ = {**DEFAULT_PROSUMER_ECONOMICS, **(prosumer_economics or {})}

        # Demand and meteo
        self.demand = np.asarray(demand, dtype=float)
        self.meteo = meteo

        # PV configuration
        self.n_series_pv = n_series_pv
        self.tilt = tilt
        self.cap_module_pv = self.pv_module['vmppt_ref'] * self.pv_module['imppt_ref'] / 1000  # kW
        self.area_module = self.pv_module['area']  # m²

        # BESS configuration
        self.cap_unit_bess = self.bess_tech['cap_unit']

        # Constraints
        self.space_max = space_max
        self.utilization_factor = utilization_factor
        self.specific_budget = specific_budget
        self.min_self_cons_ratio = min_self_cons_ratio
        self.max_prod_dem_ratio = max_prod_dem_ratio

        # --- Compute decision-variable bounds ---
        self.net_surface = space_max * utilization_factor
        self.n_parallel_pv_max = int(self.net_surface / (n_series_pv * self.area_module))
        if self.n_parallel_pv_max < 1:
            raise ValueError(
                f"Cannot fit even one PV string ({n_series_pv} modules × "
                f"{self.area_module:.2f} m² = {n_series_pv * self.area_module:.1f} m²) "
                f"in {self.net_surface:.1f} m² net surface"
            )

        kWp_max = self.n_parallel_pv_max * n_series_pv * self.cap_module_pv
        self.budget_max = specific_budget * kWp_max

        cap_bess_per_parallel = self.cap_unit_bess
        bess_cost_per_parallel = self.bess_econ['cap_cost'] * cap_bess_per_parallel
        self.n_parallel_bess_max = max(1, int(self.budget_max / bess_cost_per_parallel))

        # --- Precompute PV output for a single string ---
        self._precompute_pv()

        # Annual demand (constant across evaluations)
        self._dem_mwh = float(np.sum(self.demand)) * time_step / 1000

        # Precompute baseline NPV (no investment: all demand bought from grid)
        pe = self.pros_econ
        self._baseline_npv = 0.0
        for y in range(1, time_horizon + 1):
            annual_cost = self._dem_mwh * pe['price_buy'] * (1 - pe['decay']) ** (y - 1)
            self._baseline_npv -= annual_cost / (1 + pe['int_rate']) ** y

        # Results storage
        self.result = None
        self.pareto_X = None
        self.pareto_F = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _precompute_pv(self):
        """Compute PV production for one string (n_parallel=1), to be scaled by X[0]."""
        pv_ref = PvPanels(
            id='_pv_ref',
            n_series=self.n_series_pv,
            n_parallel=1,
            cap_cost=0, opex_cost=0,
            inc_year=0, inc_start_end=[0, 0], tax_year=0,
            **self.pv_module,
        )
        pv_ref.compute_output(
            slope=self.tilt,
            I_beam=self.meteo['I_beam'],
            I_skydiff=self.meteo['I_skydiff'],
            I_grounddiff=self.meteo['I_grounddiff'],
            t_amb=self.meteo['t_amb'],
            theta=self.meteo.get('theta'),
            wind_speed=self.meteo.get('wind_speed'),
        )
        self._prod_per_string = pv_ref.en_perf_evolution['electricity']['prod']

    def _make_bess(self, n_parallel):
        """Create a fresh Bess object for simulation."""
        no_cost = {'item1': {'unit': 0, 'cost_unit': 0, 'dur': [0, 0]}}
        no_rev = {'item1': {'unit': 0, 'rev_unit': 0, 'dur': [0, 0]}}
        total_cap = n_parallel * self.cap_unit_bess
        return Bess(
            id='_bess_opt',
            cap=total_cap,
            c_rate=self.bess_tech.get('c_rate', 1.0),
            soc_in=self.bess_tech['soc_in'],
            soc_max=self.bess_tech['soc_max'],
            soc_min=self.bess_tech['soc_min'],
            eta_charge=self.bess_tech.get('eta_charge', 0.95),
            eta_discharge=self.bess_tech.get('eta_discharge', 0.95),
            self_discharge_rate_per_hour=self.bess_tech.get('self_discharge_rate_per_hour', 0.00004),
            lifetime_years=self.bess_tech.get('lifetime_years', 15),
            # Economics params (needed by Bess.__init__ -> System.__init__)
            cap_cost=self.bess_econ['cap_cost'],
            opex_cost=self.bess_econ['opex_cost'],
            inc_year=0, inc_start_end=[0, 0], tax_year=0,
            other_cost=no_cost, other_rev=no_rev,
        )

    def _make_pv_proxy(self, cap_pv):
        """Create a lightweight PV proxy for Economics."""
        ec = self.pv_econ
        no_rev = {'item1': {'unit': 0, 'rev_unit': 0, 'dur': [0, 0]}}

        # Incentives
        if ec.get('inc_rate', 0) > 0:
            inc_total = min(ec['inc_rate'] * ec['cap_cost'] * cap_pv, ec.get('inc_cap', 1e9))
            inc_dur = ec.get('inc_duration', 10)
            inc_year = inc_total / inc_dur
            inc_start_end = [1, inc_dur]
        else:
            inc_year = ec.get('inc_year', 0)
            inc_start_end = list(ec.get('inc_start_end', [0, 0]))

        # Inverter replacement (key must match other_rev keys for Economics)
        inv_year = ec.get('inverter_year', 10)
        inv_cost = ec.get('inverter_cost', 350)
        other_cost = {'item1': {'unit': cap_pv, 'cost_unit': inv_cost, 'dur': [inv_year, inv_year]}}

        return _SystemProxy(
            id='pv_opt', cap=cap_pv, cap_cost=ec['cap_cost'],
            opex=cap_pv, opex_cost=ec['opex_cost'],
            inc_year=inc_year, inc_start_end=inc_start_end,
            tax_year=ec.get('tax_year', 0),
            other_cost=other_cost, other_rev=no_rev,
        )

    def _make_bess_proxy(self, cap_bess):
        """Create a lightweight BESS proxy for Economics."""
        ec = self.bess_econ
        no_cost = {'item1': {'unit': 0, 'cost_unit': 0, 'dur': [0, 0]}}
        no_rev = {'item1': {'unit': 0, 'rev_unit': 0, 'dur': [0, 0]}}

        if ec.get('inc_rate', 0) > 0:
            inc_total = min(ec['inc_rate'] * ec['cap_cost'] * cap_bess, ec.get('inc_cap', 1e9))
            inc_dur = ec.get('inc_duration', 10)
            inc_year = inc_total / inc_dur
            inc_start_end = [1, inc_dur]
        else:
            inc_year = ec.get('inc_year', 0)
            inc_start_end = list(ec.get('inc_start_end', [0, 0]))

        return _BessProxy(
            id='bess_opt', cap=cap_bess, cap_cost=ec['cap_cost'],
            opex=cap_bess, opex_cost=ec['opex_cost'],
            inc_year=inc_year, inc_start_end=inc_start_end,
            tax_year=ec.get('tax_year', 0),
            other_cost=no_cost, other_rev=no_rev,
            lifetime_years=self.bess_tech.get('lifetime_years', 15),
        )

    def _evaluate_individual(self, X):
        """Evaluate a single [n_parallel_pv, n_parallel_bess] configuration.

        Returns (objectives, constraints) as two lists, both in pymoo convention
        (minimise objectives, g <= 0 for feasible).
        """
        n_par_pv = int(X[0])
        n_par_bess = int(X[1])

        # --- Energy balance ---
        prod = n_par_pv * self._prod_per_string
        dem = self.demand

        self_cons_no_bess = np.minimum(prod, dem)
        surplus_no_bess = np.maximum(0.0, prod - dem)
        unmet_no_bess = np.maximum(0.0, dem - prod)

        if n_par_bess > 0:
            bess = self._make_bess(n_par_bess)
            controller = Controller(bess=[bess])
            stored, supply, _, surplus, deficit, _ = controller.energy_performance(
                production=prod, demand=dem, time=self.time_step,
            )
            self_cons_final = self_cons_no_bess + stored
            surplus_final = surplus
            unmet_final = deficit
        else:
            self_cons_final = self_cons_no_bess
            surplus_final = surplus_no_bess
            unmet_final = unmet_no_bess

        # --- Annual energy flows (MWh) ---
        ts = self.time_step
        self_cons_mwh = float(np.sum(self_cons_final)) * ts / 1000
        surplus_mwh = float(np.sum(surplus_final)) * ts / 1000
        purchased_mwh = float(np.sum(unmet_final)) * ts / 1000
        prod_mwh = float(np.sum(prod)) * ts / 1000
        dem_mwh = self._dem_mwh

        # --- Capacities ---
        cap_pv = n_par_pv * self.n_series_pv * self.cap_module_pv      # kW
        cap_bess = n_par_bess * self.n_series_bess * self.cap_module_bess  # kWh

        # --- Economic evaluation ---
        components = [self._make_pv_proxy(cap_pv)]
        if n_par_bess > 0:
            components.append(self._make_bess_proxy(cap_bess))

        pe = self.pros_econ
        flows = {
            'electricity': {
                'sold': surplus_mwh,
                'self_cons': self_cons_mwh,
                'purchased': purchased_mwh,
                'price_sold': pe['price_sold'],
                'price_buy': pe['price_buy'],
                'decay': pe['decay'],
                'prod_degradation': pe.get('prod_degradation', 0.0),
            }
        }

        ec_perf = Economics(
            components=components,
            annual_en_flows_and_prices=flows,
        ).compute_cashflow(
            time_horizon=self.time_horizon,
            tax_rate=pe['tax_rate'],
            int_rate=pe['int_rate'],
        )
        # Incremental NPV = NPV(with investment) - NPV(without investment)
        # A positive value means the investment is financially beneficial.
        npv = ec_perf['NPV'] - self._baseline_npv

        # --- Objectives (pymoo minimises) ---
        if self.objective_mode == 'self_consumption':
            f1 = -self_cons_mwh
        else:
            self_suff = (dem_mwh - purchased_mwh) / dem_mwh if dem_mwh > 0 else 0.0
            f1 = -self_suff
        f2 = -npv

        # --- Constraints (g <= 0 is feasible) ---
        ratio_prod_dem = prod_mwh / dem_mwh if dem_mwh > 0 else 0.0
        ratio_self_cons = self_cons_mwh / dem_mwh if dem_mwh > 0 else 0.0

        capex_total = self.pv_econ['cap_cost'] * cap_pv
        if n_par_bess > 0:
            capex_total += self.bess_econ['cap_cost'] * cap_bess

        g1 = ratio_prod_dem - self.max_prod_dem_ratio        # prod/dem <= limit
        g2 = self.min_self_cons_ratio - ratio_self_cons       # self_cons/dem >= limit
        g3 = capex_total - self.budget_max                    # CAPEX <= budget

        return [f1, f2], [g1, g2, g3]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def optimize(self):
        """Run the NSGA-II optimisation.

        Returns
        -------
        pd.DataFrame
            Pareto-optimal solutions with design variables, capacities,
            CAPEX and objective values.
        """
        optimizer_ref = self

        class _Problem(ElementwiseProblem):
            def __init__(self):
                super().__init__(
                    n_var=2,
                    n_obj=2,
                    n_constr=3,
                    xl=np.array([1, 0], dtype=int),
                    xu=np.array([optimizer_ref.n_parallel_pv_max,
                                 optimizer_ref.n_parallel_bess_max], dtype=int),
                    type_var=int,
                )

            def _evaluate(self, X, out, *args, **kwargs):
                F, G = optimizer_ref._evaluate_individual(X)
                out["F"] = F
                out["G"] = G

        algorithm = NSGA2(
            pop_size=self.pop_size,
            n_offsprings=self.pop_size // 2,
            sampling=IntegerRandomSampling(),
            crossover=SBX(prob=0.9, eta=15, repair=None),
            mutation=PM(prob=0.5, eta=20),
            eliminate_duplicates=True,
        )

        termination = TerminationCollection(
            DefaultMultiObjectiveTermination(
                ftol=0.005, xtol=0.0005, n_skip=5, period=15,
            ),
            get_termination("n_gen", self.n_gen),
        )

        self.result = minimize(
            _Problem(),
            algorithm,
            termination,
            seed=self.seed,
            save_history=True,
            verbose=True,
        )

        if self.result.X is not None:
            self.pareto_X = np.atleast_2d(self.result.X)
            self.pareto_F = -np.atleast_2d(self.result.F)
        else:
            raise RuntimeError("Optimisation found no feasible solutions. "
                               "Relax constraints or increase budget/space.")

        return self.get_results()

    def get_results(self):
        """Return Pareto-front solutions as a DataFrame.

        Columns: n_parallel_pv, n_parallel_bess, cap_pv_kW, cap_bess_kWh,
                 capex_EUR, <objective_1>, NPV_EUR.
        """
        if self.pareto_X is None:
            raise RuntimeError("Call optimize() first")

        X = self.pareto_X
        F = self.pareto_F

        cap_pv = X[:, 0] * self.n_series_pv * self.cap_module_pv
        cap_bess = X[:, 1] * self.n_series_bess * self.cap_module_bess
        capex = (cap_pv * self.pv_econ['cap_cost']
                 + cap_bess * self.bess_econ['cap_cost'])

        obj1_label = ('self_cons_MWh' if self.objective_mode == 'self_consumption'
                       else 'self_sufficiency')

        df = pd.DataFrame({
            'n_parallel_pv': X[:, 0].astype(int),
            'n_parallel_bess': X[:, 1].astype(int),
            'cap_pv_kW': np.round(cap_pv, 2),
            'cap_bess_kWh': np.round(cap_bess, 2),
            'capex_EUR': np.round(capex, 0),
            obj1_label: np.round(F[:, 0], 4),
            'NPV_EUR': np.round(F[:, 1], 0),
        })

        return df.sort_values(obj1_label, ascending=False).reset_index(drop=True)

    def plot_pareto(self, figsize=(14, 5)):
        """Plot the Pareto front coloured by PV capacity and BESS capacity.

        Returns
        -------
        matplotlib.figure.Figure
        """
        if self.pareto_X is None:
            raise RuntimeError("Call optimize() first")

        X = self.pareto_X
        F = self.pareto_F

        cap_pv = X[:, 0] * self.n_series_pv * self.cap_module_pv
        cap_bess = X[:, 1] * self.n_series_bess * self.cap_module_bess

        xlabel = ('Self-consumption [MWh/y]' if self.objective_mode == 'self_consumption'
                  else 'Self-sufficiency rate [-]')

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        sc1 = ax1.scatter(F[:, 0], F[:, 1] / 1000, s=60, c=cap_pv,
                          cmap='rainbow', edgecolors='k', linewidths=0.5)
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel('NPV [k€]')
        cb1 = fig.colorbar(sc1, ax=ax1)
        cb1.set_label('PV [kWp]')
        ax1.grid(True, alpha=0.3)

        sc2 = ax2.scatter(F[:, 0], F[:, 1] / 1000, s=60, c=cap_bess,
                          cmap='rainbow', edgecolors='k', linewidths=0.5)
        ax2.set_xlabel(xlabel)
        ax2.set_ylabel('NPV [k€]')
        cb2 = fig.colorbar(sc2, ax=ax2)
        cb2.set_label('BESS [kWh]')
        ax2.grid(True, alpha=0.3)

        fig.suptitle(f'Pareto Front — objective: {self.objective_mode}', fontsize=13)
        fig.tight_layout()
        return fig

    def plot_convergence(self, figsize=(8, 5)):
        """Plot normalised hypervolume convergence across generations.

        Returns
        -------
        matplotlib.figure.Figure
        """
        if self.result is None or not self.result.history:
            raise RuntimeError("Call optimize() first")

        from pymoo.indicators.hv import Hypervolume

        F_all = np.vstack([gen.pop.get("F") for gen in self.result.history])
        ref_point = np.max(F_all, axis=0) + 1
        hv = Hypervolume(ref_point=ref_point)

        hv_values, n_evals, count = [], [], 0
        for gen in self.result.history:
            hv_values.append(hv.do(gen.pop.get("F")))
            count += len(gen.pop)
            n_evals.append(count)

        hv_arr = np.array(hv_values)

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(n_evals, hv_arr / hv_arr.max(), marker='o', markersize=4)
        ax.set_xlabel('Function Evaluations')
        ax.set_ylabel('Normalised Hypervolume')
        ax.set_title('NSGA-II Convergence')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig

    def simulate_solution(self, n_parallel_pv, n_parallel_bess):
        """Run a detailed simulation for a specific PV+BESS configuration.

        Returns a dict with full time-series and economic results, suitable
        for plotting energy balance, cashflow, and environmental analysis.
        """
        prod = n_parallel_pv * self._prod_per_string
        dem = self.demand

        self_cons_no_bess = np.minimum(prod, dem)
        surplus_no_bess = np.maximum(0.0, prod - dem)
        unmet_no_bess = np.maximum(0.0, dem - prod)

        energy = {'prod': prod, 'dem': dem}

        if n_parallel_bess > 0:
            bess = self._make_bess(n_parallel_bess)
            controller = Controller(bess=[bess])
            stored, supply, power, surplus, deficit, soc = controller.energy_performance(
                production=prod, demand=dem, time=self.time_step)
            energy['self_cons'] = self_cons_no_bess + stored
            energy['surplus'] = surplus
            energy['unmet'] = deficit
            energy['stored'] = stored
            energy['supply'] = supply
            energy['soc'] = soc
        else:
            energy['self_cons'] = self_cons_no_bess
            energy['surplus'] = surplus_no_bess
            energy['unmet'] = unmet_no_bess
            energy['stored'] = np.zeros_like(dem)
            energy['supply'] = np.zeros_like(dem)
            energy['soc'] = np.zeros_like(dem)

        ts = self.time_step
        annual = {
            'prod_MWh': float(np.sum(prod)) * ts / 1000,
            'dem_MWh': self._dem_mwh,
            'self_cons_MWh': float(np.sum(energy['self_cons'])) * ts / 1000,
            'surplus_MWh': float(np.sum(energy['surplus'])) * ts / 1000,
            'purchased_MWh': float(np.sum(energy['unmet'])) * ts / 1000,
        }
        annual['self_suff_rate'] = ((annual['dem_MWh'] - annual['purchased_MWh'])
                                    / annual['dem_MWh']) if annual['dem_MWh'] > 0 else 0
        annual['self_cons_rate'] = (annual['self_cons_MWh']
                                    / annual['prod_MWh']) if annual['prod_MWh'] > 0 else 0

        cap_pv = n_parallel_pv * self.n_series_pv * self.cap_module_pv
        cap_bess = n_parallel_bess * self.n_series_bess * self.cap_module_bess

        components = [self._make_pv_proxy(cap_pv)]
        if n_parallel_bess > 0:
            components.append(self._make_bess_proxy(cap_bess))

        pe = self.pros_econ
        flows = {
            'electricity': {
                'sold': annual['surplus_MWh'],
                'self_cons': annual['self_cons_MWh'],
                'purchased': annual['purchased_MWh'],
                'price_sold': pe['price_sold'],
                'price_buy': pe['price_buy'],
                'decay': pe['decay'],
                'prod_degradation': pe.get('prod_degradation', 0.0),
            }
        }
        ec_perf = Economics(
            components=components, annual_en_flows_and_prices=flows,
        ).compute_cashflow(
            time_horizon=self.time_horizon, tax_rate=pe['tax_rate'], int_rate=pe['int_rate'])
        ec_perf['NPV_incremental'] = ec_perf['NPV'] - self._baseline_npv

        config = {
            'n_parallel_pv': n_parallel_pv, 'n_parallel_bess': n_parallel_bess,
            'cap_pv_kW': cap_pv, 'cap_bess_kWh': cap_bess, 'capex_EUR': ec_perf['capex'],
        }
        return {'energy': energy, 'annual': annual, 'economics': ec_perf, 'config': config}

    def summary(self):
        """Print a summary of the optimiser configuration."""
        cap_str_pv = self.n_series_pv * self.cap_module_pv
        print(f"{'='*60}")
        print(f"  ProsumerOptimizer — {self.objective_mode}")
        print(f"{'='*60}")
        print(f"  Time step:           {self.time_step} h  ({len(self.demand)} steps)")
        print(f"  Annual demand:       {self._dem_mwh:.2f} MWh")
        print(f"  PV module:           {self.cap_module_pv*1000:.0f} W, {self.area_module:.2f} m²")
        print(f"  PV string:           {self.n_series_pv} modules → {cap_str_pv:.2f} kWp")
        print(f"  Net surface:         {self.net_surface:.1f} m²")
        print(f"  Max PV strings:      {self.n_parallel_pv_max}  ({self.n_parallel_pv_max * cap_str_pv:.1f} kWp)")
        print(f"  BESS module:         {self.cap_module_bess:.3f} kWh")
        print(f"  Max BESS parallel:   {self.n_parallel_bess_max}  ({self.n_parallel_bess_max * self.n_series_bess * self.cap_module_bess:.1f} kWh)")
        print(f"  Budget max:          {self.budget_max:,.0f} €")
        print(f"  NSGA-II:             pop={self.pop_size}, gen={self.n_gen}")
        print(f"{'='*60}")
