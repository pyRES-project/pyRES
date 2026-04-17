"""
Multi-objective sizing optimizer for a prosumer (production system + BESS)
using NSGA-II.

The optimizer works with ready-made pyRES objects: the caller builds a
`System`-like production object and a `Bess` object as *unit templates*
(attributes `cap`, `cap_cost_unit`, economics, etc. describe ONE unit),
provides a normalized production curve (kW per kWp), a demand curve and
economic parameters, and chooses a sizing `mode`.

Modes
-----
'system_and_bess' : decision variables are [n_parallel_system, n_parallel_bess].
                    Total system capacity = n_parallel_system * system.cap,
                    scaled production = total_kWp * prod_per_kwp.
'bess_only'       : decision variable is [n_parallel_bess].
                    System is used as given (fixed cap and fixed production).

Objectives (both minimised internally; NSGA-II convention):
    f1 = -self_consumption [MWh/y]   if objective_mode='self_consumption'
         -self_sufficiency [-]       if objective_mode='self_sufficiency'
    f2 = -NPV [EUR]                  (incremental w.r.t. baseline "all from grid")

Constraints (g <= 0 is feasible):
    g1: prod / dem <= max_prod_dem_ratio
    g2: self_cons / dem >= min_self_cons_ratio
    g3: CAPEX <= budget_max           (only if budget_max is provided)

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

from src.rec_sim.Bess import Bess
from src.rec_sim.Consumer import Consumer
from src.rec_sim.Prosumer import Prosumer
from src.rec_sim.System import System


DEFAULT_ECONOMICS = {
    'tax_rate': 0.2,
    'int_rate': 0.05,
    'price_buy': 250,              # €/MWh
    'price_sold': 104,             # €/MWh
    'decay': 0.02,                 # annual fractional decay of energy prices
    'other_capex_perc': [0],       # list passed to Prosumer.economic_performance
}


# ---------------------------------------------------------------------------
# Main optimizer class
# ---------------------------------------------------------------------------
# The production subclass (e.g. PvPanels) is used only once, by the caller,
# to shape the template. Inside the NSGA-II loop we instantiate the base
# `System` class directly — its __init__ is cheap — and attach the scaled
# production curve to `en_perf_evolution` so `Prosumer.energy_performance`
# can consume it like any real production component.
class ProsumerOptimizer:
    """Multi-objective sizing optimizer for a prosumer."""

    def __init__(self,
                 system, bess,
                 demand, prod_per_kwp,
                 economics,
                 time_step, time_horizon,
                 mode='system_and_bess',
                 objective_mode='self_consumption',
                 n_system_max=None,
                 n_bess_max=None,
                 budget_max=None,
                 min_self_cons_ratio=0.0,
                 max_prod_dem_ratio=10.0,
                 pop_size=100,
                 n_gen=50,
                 seed=42):
        """
        Parameters
        ----------
        system : System-like
            Production system (e.g. PvPanels) representing ONE unit. Its
            `cap` and economic attributes are used as the unit template;
            the optimizer scales capacity by integer n_parallel.
        bess : Bess
            Battery representing ONE unit. Same convention as `system`.
        demand : array-like
            Load profile in kW.
        prod_per_kwp : array-like
            Production profile normalized to 1 kWp of installed system [kW/kWp],
            same length and time step as `demand`.
        economics : dict
            Prosumer economics: keys `tax_rate`, `int_rate`, `price_buy`,
            `price_sold`, `decay`, optional `other_capex_perc`.
        time_step : float
            Time step in hours (e.g. 0.25 for 15 min).
        time_horizon : int
            Investment horizon in years.
        mode : {'system_and_bess', 'bess_only'}
            Selects the set of decision variables.
        objective_mode : {'self_consumption', 'self_sufficiency'}
            Alternative first objective.
        n_system_max : int, optional
            Max integer multiple of the system template. Required in
            'system_and_bess' mode.
        n_bess_max : int, optional
            Max integer multiple of the BESS template. Required.
        budget_max : float, optional
            Hard upper bound on CAPEX [€]. If None the budget constraint is dropped.
        min_self_cons_ratio, max_prod_dem_ratio : float
            KPI-based feasibility constraints.
        """
        if mode not in ('system_and_bess', 'bess_only'):
            raise ValueError(f"mode must be 'system_and_bess' or 'bess_only', got '{mode}'")
        if objective_mode not in ('self_consumption', 'self_sufficiency'):
            raise ValueError(f"objective_mode must be 'self_consumption' or 'self_sufficiency', "
                             f"got '{objective_mode}'")
        if mode == 'system_and_bess' and n_system_max is None:
            raise ValueError("n_system_max is required in 'system_and_bess' mode")
        if n_bess_max is None:
            raise ValueError("n_bess_max is required")

        self.system_template = system
        self.bess_template = bess
        self.demand = np.asarray(demand, dtype=float)
        self.prod_per_kwp = np.asarray(prod_per_kwp, dtype=float)
        self.economics = {**DEFAULT_ECONOMICS, **economics}
        self.time_step = time_step
        self.time_horizon = time_horizon
        self.mode = mode
        self.objective_functions = objective_mode

        self.n_system_max = int(n_system_max) if n_system_max is not None else None
        self.n_bess_max = int(n_bess_max)
        self.budget_max = budget_max
        self.min_self_cons_ratio = min_self_cons_ratio
        self.max_prod_dem_ratio = max_prod_dem_ratio

        self.pop_size = pop_size
        self.n_gen = n_gen
        self.seed = seed

        self.consumer = Consumer(id='c_opt', dem={'electricity': self.demand})

        # Baseline NPV: no investment, all demand purchased from grid
        self.dem_mwh = float(np.sum(self.demand)) * time_step / 1000
        ec = self.economics
        self.baseline_npv = 0.0
        for y in range(1, time_horizon + 1):
            annual_cost = self.dem_mwh * ec['price_buy'] * (1 - ec['decay']) ** (y - 1)
            self.baseline_npv -= annual_cost / (1 + ec['int_rate']) ** y

        self.result = None
        self.pareto_X = None
        self.pareto_F = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_bess(self, n_modules):
        """Build a Bess instance scaled by n_parallel, copying the template's params."""
        t = self.bess_template
        return Bess(
            id=t.id,
            cap=n_modules * t.cap,
            c_rate=t.c_rate,
            soc_in=t.soc_in, soc_max=t.soc_max, soc_min=t.soc_min,
            eta_charge=t.eta_charge, eta_discharge=t.eta_discharge,
            self_discharge_rate_per_hour=t.self_discharge_rate_per_hour,
            lifetime_years=t.lifetime_years,
            carriers=t.carriers,
            cap_cost=t.cap_cost_unit,
            opex_cost=t.opex_cost,
            inc_year=t.inc_year,
            inc_start_end=t.inc_start_end,
            tax_year=t.tax_year,
            other_cost=t.other_cost,
            other_rev=t.other_rev,
        )

    def _make_system(self, n_module):
        """Build a scaled System with its production curve attached.

        In 'bess_only' mode n_parallel is forced to 1 so the template is used as-is.
        """
        if self.mode == 'bess_only':
            n_module = 1
        t = self.system_template
        cap = n_module * t.cap
        sys = System(
            id=t.id, carriers=t.carriers,
            cap=cap, cap_cost=t.cap_cost_unit,
            opex=cap, opex_cost=t.opex_cost,
            inc_year=t.inc_year, inc_start_end=t.inc_start_end,
            tax_year=t.tax_year,
            other_cost=t.other_cost, other_rev=t.other_rev,
        )
        carrier = t.carriers[0]
        sys.en_perf_evolution[carrier] = {'prod': cap * self.prod_per_kwp}
        return sys

    def _build_prosumer(self, n_system, n_bess):
        """Assemble a real Prosumer for the given candidate."""
        sys = self._make_system(n_system if self.mode == 'system_and_bess' else 1)
        bess_list = [self._make_bess(n_bess)] if n_bess > 0 else []
        return Prosumer(
            id='p_opt', carriers=['electricity'],
            systems=[sys], users=[self.consumer], bess=bess_list,
        )

    def _flows_from_prosumer(self, prosumer):
        """Build the en_flows_and_prices dict from the computed annual flows."""
        ec = self.economics
        flows = {}
        for carrier in prosumer.carriers:
            ann = prosumer.en_perf_evolution[carrier]['annual']
            flows[carrier] = {
                'sold': ann['surplus'],
                'self_cons': ann['self_cons'],
                'purchased': ann['unmet'],
                'price_sold': ec['price_sold'],
                'price_buy': ec['price_buy'],
                'decay': ec['decay'],
            }
        return flows

    def _decode(self, X):
        """Return (n_system, n_bess) from a candidate vector."""
        if self.mode == 'system_and_bess':
            return int(X[0]), int(X[1])
        return None, int(X[0])

    def _evaluate_individual(self, X):
        """Evaluate a candidate. Returns (objectives, constraints) per pymoo."""
        n_system, n_bess = self._decode(X)
        ec = self.economics

        prosumer = self._build_prosumer(n_system, n_bess)
        prosumer.energy_performance(self.time_step)
        prosumer.economic_performance(
            time_horizon=self.time_horizon,
            tax_rate=ec['tax_rate'], int_rate=ec['int_rate'],
            other_capex_perc=ec['other_capex_perc'],
            en_flows_and_prices=self._flows_from_prosumer(prosumer),
        )

        ann = prosumer.en_perf_evolution['electricity']['annual']
        prod_mwh = ann['prod']
        self_cons_mwh = ann['self_cons']
        purchased_mwh = ann['unmet']
        dem_mwh = ann['dem']
        npv = prosumer.ec_perf['NPV'] - self.baseline_npv

        if self.objective_functions == 'self_consumption':
            f1 = -self_cons_mwh
        else:
            self_suff = (dem_mwh - purchased_mwh) / dem_mwh if dem_mwh > 0 else 0.0
            f1 = -self_suff
        f2 = -npv

        ratio_prod_dem = prod_mwh / dem_mwh if dem_mwh > 0 else 0.0
        ratio_self_cons = self_cons_mwh / dem_mwh if dem_mwh > 0 else 0.0

        cap_system = prosumer.systems[0].cap
        capex_total = self.system_template.cap_cost_unit * cap_system
        if prosumer.bess:
            capex_total += self.bess_template.cap_cost_unit * prosumer.bess[0].cap

        g1 = ratio_prod_dem - self.max_prod_dem_ratio
        g2 = self.min_self_cons_ratio - ratio_self_cons
        if self.budget_max is not None:
            g3 = capex_total - self.budget_max
        else:
            g3 = -1.0  # always feasible

        return [f1, f2], [g1, g2, g3]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def optimize(self):
        """Run the NSGA-II optimisation and return the Pareto-front DataFrame."""
        optimizer_ref = self

        if self.mode == 'system_and_bess':
            n_var = 2
            xl = np.array([1, 0], dtype=int)
            xu = np.array([self.n_system_max, self.n_bess_max], dtype=int)
        else:
            n_var = 1
            xl = np.array([0], dtype=int)
            xu = np.array([self.n_bess_max], dtype=int)

        class _Problem(ElementwiseProblem):
            def __init__(self):
                super().__init__(
                    n_var=n_var, n_obj=2, n_constr=3,
                    xl=xl, xu=xu, type_var=int,
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
            _Problem(), algorithm, termination,
            seed=self.seed, save_history=True, verbose=True,
        )

        if self.result.X is None:
            raise RuntimeError(
                "Optimisation found no feasible solutions. "
                "Relax constraints or increase budget/max sizes."
            )

        self.pareto_X = np.atleast_2d(self.result.X)
        self.pareto_F = -np.atleast_2d(self.result.F)
        return self.get_results()

    def _capacities_from_X(self, X):
        """Return (cap_system_kW, cap_bess_kWh) arrays from the Pareto X matrix."""
        if self.mode == 'system_and_bess':
            cap_system = X[:, 0] * self.system_template.cap
            cap_bess = X[:, 1] * self.bess_template.cap
        else:
            cap_system = np.full(X.shape[0], self.system_template.cap)
            cap_bess = X[:, 0] * self.bess_template.cap
        return cap_system, cap_bess

    def get_results(self):
        """Return Pareto-front solutions as a DataFrame."""
        if self.pareto_X is None:
            raise RuntimeError("Call optimize() first")

        X = self.pareto_X
        F = self.pareto_F
        cap_system, cap_bess = self._capacities_from_X(X)
        capex = (cap_system * self.system_template.cap_cost_unit
                 + cap_bess * self.bess_template.cap_cost_unit)

        obj1_label = ('self_cons_MWh' if self.objective_functions == 'self_consumption'
                      else 'self_sufficiency')

        cols = {}
        if self.mode == 'system_and_bess':
            cols['n_system'] = X[:, 0].astype(int)
            cols['n_bess'] = X[:, 1].astype(int)
        else:
            cols['n_bess'] = X[:, 0].astype(int)
        cols['cap_system_kW'] = np.round(cap_system, 2)
        cols['cap_bess_kWh'] = np.round(cap_bess, 2)
        cols['capex_EUR'] = np.round(capex, 0)
        cols[obj1_label] = np.round(F[:, 0], 4)
        cols['NPV_EUR'] = np.round(F[:, 1], 0)

        df = pd.DataFrame(cols)
        return df.sort_values(obj1_label, ascending=False).reset_index(drop=True)

    def plot_pareto(self, figsize=(14, 5)):
        """Plot the Pareto front coloured by system and BESS capacities."""
        if self.pareto_X is None:
            raise RuntimeError("Call optimize() first")

        X = self.pareto_X
        F = self.pareto_F
        cap_system, cap_bess = self._capacities_from_X(X)

        xlabel = ('Self-consumption [MWh/y]' if self.objective_functions == 'self_consumption'
                  else 'Self-sufficiency rate [-]')

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        sc1 = ax1.scatter(F[:, 0], F[:, 1] / 1000, s=60, c=cap_system,
                          cmap='rainbow', edgecolors='k', linewidths=0.5)
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel('NPV [k€]')
        fig.colorbar(sc1, ax=ax1).set_label('System [kW]')
        ax1.grid(True, alpha=0.3)

        sc2 = ax2.scatter(F[:, 0], F[:, 1] / 1000, s=60, c=cap_bess,
                          cmap='rainbow', edgecolors='k', linewidths=0.5)
        ax2.set_xlabel(xlabel)
        ax2.set_ylabel('NPV [k€]')
        fig.colorbar(sc2, ax=ax2).set_label('BESS [kWh]')
        ax2.grid(True, alpha=0.3)

        fig.suptitle(f'Pareto Front — objective: {self.objective_functions}', fontsize=13)
        fig.tight_layout()
        return fig

    def plot_convergence(self, figsize=(8, 5)):
        """Plot normalised hypervolume convergence across generations."""
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

    def simulate_solution(self, n_system=None, n_bess=0):
        """Run a detailed simulation for a specific configuration.

        In 'bess_only' mode `n_system` is ignored (the template is used as-is).
        """
        ec = self.economics
        prosumer = self._build_prosumer(n_system, n_bess)
        prosumer.energy_performance(self.time_step)
        ec_perf = prosumer.economic_performance(
            time_horizon=self.time_horizon,
            tax_rate=ec['tax_rate'], int_rate=ec['int_rate'],
            other_capex_perc=ec['other_capex_perc'],
            en_flows_and_prices=self._flows_from_prosumer(prosumer),
        )
        ec_perf['NPV_incremental'] = ec_perf['NPV'] - self.baseline_npv

        ep = prosumer.en_perf_evolution['electricity']
        ann = ep['annual']
        energy = {
            'prod': ep['prod'], 'dem': ep['dem'],
            'self_cons': ep['self_cons'], 'surplus': ep['surplus'], 'unmet': ep['unmet'],
            'power_from_source': ep.get('power_from_source', np.zeros_like(ep['dem'])),
            'supply': ep.get('supply', np.zeros_like(ep['dem'])),
            'soc': ep.get('soc', np.zeros_like(ep['dem'])),
        }
        annual = {
            'prod_MWh': ann['prod'], 'dem_MWh': ann['dem'],
            'self_cons_MWh': ann['self_cons'], 'surplus_MWh': ann['surplus'],
            'purchased_MWh': ann['unmet'],
            'self_suff_rate': ann['ss_rate'] / 100,
            'self_cons_rate': ann['sc_rate'] / 100,
        }
        config = {
            'n_system': n_system, 'n_bess': n_bess,
            'cap_system_kW': prosumer.systems[0].cap,
            'cap_bess_kWh': prosumer.bess[0].cap if prosumer.bess else 0.0,
            'capex_EUR': ec_perf['capex'],
        }
        return {'energy': energy, 'annual': annual, 'economics': ec_perf, 'config': config}

    def summary(self):
        """Print a short summary of the optimiser configuration."""
        print('=' * 60)
        print(f'  ProsumerOptimizer — mode={self.mode}, obj={self.objective_functions}')
        print('=' * 60)
        print(f'  Time step:           {self.time_step} h  ({len(self.demand)} steps)')
        print(f'  Annual demand:       {self.dem_mwh:.2f} MWh')
        print(f'  System unit cap:     {self.system_template.cap:.3f} kW')
        if self.mode == 'system_and_bess':
            print(f'  n_system_max:        {self.n_system_max}  '
                  f'({self.n_system_max * self.system_template.cap:.1f} kW)')
        print(f'  BESS unit cap:       {self.bess_template.cap:.3f} kWh')
        print(f'  n_bess_max:          {self.n_bess_max}  '
              f'({self.n_bess_max * self.bess_template.cap:.1f} kWh)')
        if self.budget_max is not None:
            print(f'  Budget max:          {self.budget_max:,.0f} €')
        print(f'  NSGA-II:             pop={self.pop_size}, gen={self.n_gen}')
        print('=' * 60)
