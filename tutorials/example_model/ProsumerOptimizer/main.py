"""
Created on April 21 08:00:00 2026

@author: isabella pizzuti
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
sys.path.insert(0, str(ROOT))


import matplotlib.pyplot as plt
import pandas as pd

from src.rec_sim.System import System
from src.rec_sim.Bess import Bess
from src.optimizer.optimizer_prosumer import ProsumerOptimizer

# --- Set input ---
time_step = 0.25
time_horizon = 20
n_system_max = 20
n_bess_max = 10
pop_size = 40
mode = 'system_and_bess'
objective_function = 'self_consumption'
n_gen = 15

pv = {'cap': 1, 'cap_cost': 1000, 'opex': 1.0, 'opex_cost': 40,
      'inc_year': 0, 'inc_start_end': [0, 0], 'tax_year': 0}  # 1 kWp per unit

bess = {'cap': 2.56,
        'c_rate': 1.0,
        'soc_in': 0.2, 'soc_max': 0.8, 'soc_min': 0.2,
        'eta_charge': 0.95, 'eta_discharge': 0.95,
        'self_discharge_rate_per_hour': 4e-5,
        'lifetime_years': 15,
        'cap_cost': 450, 'opex_cost': 20,
        'inc_year': 0, 'inc_start_end': [0, 0], 'tax_year': 0}  # 2.56 kWh per unit

prosumer_economics = {
    'tax_rate': 0.2,
    'int_rate': 0.05,
    'price_buy': 250,
    'price_sold': 104,
    'decay': 0.02,
    'other_capex_perc': [0],
}

# --- import demand curve and normalized production curve---
df = pd.read_csv('Input/prod_and_dem_kW.csv', delimiter=';')
prod_per_kwp = df['prod_per_kwp']
demand = df['demand']

# --- Templates ---
system_template = System(
    id='pv_unit', carriers=['electricity'],
    cap=pv['cap'],
    cap_cost=pv['cap_cost'],
    opex=pv['opex'], opex_cost=pv['opex_cost'],
    inc_year=pv['inc_year'], inc_start_end=pv['inc_start_end'], tax_year=pv['tax_year'],
)

bess_template = Bess(
    id='bess_unit',
    cap=bess['cap'],
    c_rate=bess['c_rate'],
    soc_in=bess['soc_in'], soc_max=bess['soc_max'], soc_min=bess['soc_min'],
    eta_charge=bess['eta_charge'], eta_discharge=bess['eta_discharge'],
    self_discharge_rate_per_hour=bess['self_discharge_rate_per_hour'],
    lifetime_years=bess['lifetime_years'],
    cap_cost=bess['cap_cost'], opex_cost=bess['opex_cost'],
    inc_year=bess['inc_year'], inc_start_end=bess['inc_start_end'], tax_year=bess['tax_year']
)

# --- Optimizer ---
optimizer = ProsumerOptimizer(
    system=system_template,
    bess=bess_template,
    demand=demand,
    prod_per_kwp=prod_per_kwp,
    economics=prosumer_economics,
    time_step=time_step,
    time_horizon=time_horizon,
    mode=mode,
    objective_function=objective_function,
    n_system_max=n_system_max,
    n_bess_max=n_bess_max,
    pop_size=pop_size,
    n_gen=n_gen,
)

optimizer.summary()
results = optimizer.optimize()
print('\nPareto front:')
print(results.to_string(index=False))
fig_pareto = optimizer.plot_pareto()
fig_convergence = optimizer.plot_convergence()
output_dir = HERE / 'Output'
output_dir.mkdir(exist_ok=True)
fig_pareto.savefig(output_dir / 'pareto.png')
fig_convergence.savefig(output_dir / 'convergence.png')
plt.show()
