"""
Minimal smoke-test runner for ProsumerOptimizer.

Builds a toy setup (1 kWp PV unit, 2.56 kWh BESS unit, flat demand, synthetic
production curve) and runs a quick NSGA-II optimisation to verify the end-to-end
pipeline works.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt

from src.rec_sim.System import System
from src.rec_sim.Bess import Bess
from src.optimizer.optimizer_prosumer import ProsumerOptimizer


def main():
    # --- Time axis: 1 year at 15-min step ---
    time_step = 0.25  # hours
    n = int(365 * 24 / time_step)
    t_hours = np.arange(n) * time_step

    # --- Synthetic per-kWp production: clipped sine, zero at night ---
    hour_of_day = t_hours % 24
    prod_per_kwp = np.clip(np.sin((hour_of_day - 6) * np.pi / 12), 0, 1) * 0.8

    # --- Flat demand 1 kW ---
    demand = np.full(n, 1.0)

    # --- Templates (one unit each) ---
    system_template = System(
        id='pv_unit', carriers=['electricity'],
        cap=1.0,              # 1 kWp per unit
        cap_cost=1000,
        opex=1.0, opex_cost=40,
        inc_year=0, inc_start_end=[0, 0], tax_year=0,
    )

    bess_template = Bess(
        id='bess_unit',
        cap=2.56,             # 2.56 kWh per unit
        c_rate=1.0,
        soc_in=0.2, soc_max=0.8, soc_min=0.2,
        eta_charge=0.95, eta_discharge=0.95,
        self_discharge_rate_per_hour=4e-5,
        lifetime_years=15,
        cap_cost=450, opex_cost=20,
        inc_year=0, inc_start_end=[0, 0], tax_year=0,
    )

    economics = {
        'tax_rate': 0.2,
        'int_rate': 0.05,
        'price_buy': 250,
        'price_sold': 104,
        'decay': 0.02,
    }

    # --- Optimizer ---
    optimizer = ProsumerOptimizer(
        system=system_template,
        bess=bess_template,
        demand=demand,
        prod_per_kwp=prod_per_kwp,
        economics=economics,
        time_step=time_step,
        time_horizon=20,
        mode='system_and_bess',
        objective_mode='self_consumption',
        n_system_max=20,
        n_bess_max=10,
        pop_size=40,
        n_gen=15,
    )

    optimizer.summary()
    results = optimizer.optimize()
    print('\nPareto front:')
    print(results.to_string(index=False))

    optimizer.plot_pareto()
    optimizer.plot_convergence()
    plt.show()


if __name__ == '__main__':
    main()
