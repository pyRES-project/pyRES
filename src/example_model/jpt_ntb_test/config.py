"""
Created on June 7 08:00:00 2025

@author: isabella pizzuti
"""

import yaml

demand_file = "Input/demand_qc_kw_el.csv"
simulation_dict = {
    "time_step": '15min', "start_date": '01-01-2020', 'time_horizon': 20, "demand_curve_file": demand_file

}

n_users = 52
base_column_name = "user"
users_dict = {}
for i in range(n_users):
    users_dict[f"user{i}"] = {
        "id": f"{i}",
        "carriers": {
            "electricity": {
                "column": f"user{i}"
            }
        }
    }

consumers_list = [f"user{i}" for i in range(2, n_users)]

# [MIGLIORAMENTO #2] Parametri elettrici del modulo PV configurabili:
# nel codice originale questi parametri erano hardcoded nel costruttore di PvPanels.
# Ora sono esplicitamente definiti nella config per ogni impianto PV, permettendo
# di simulare moduli diversi (es. monocristallino vs thin-film).
# I valori di default corrispondono al modulo NeON 2 LG370Q1C-V5 (370W, 72 celle)
pv_module_params = {
    "isc_ref": 10.47,       # (A) corrente di corto circuito a STC
    "voc_ref": 49.3,        # (V) tensione a circuito aperto a STC
    "vmppt_ref": 40.6,      # (V) tensione al punto di massima potenza a STC
    "imppt_ref": 9.86,      # (A) corrente al punto di massima potenza a STC
    "mu_isc_ref": 0.02,     # (A/K) coefficiente di temperatura di Isc
    "mu_voc_ref": 0.26,     # (V/K) coefficiente di temperatura di Voc
    "ser_cell": 60,         # numero di celle in serie nel modulo
    "t_cell_noct_c": 42,    # (°C) temperatura della cella a NOCT
    "area": 2.07,           # (m^2) area del modulo
    # [MIGLIORAMENTO #9] Bandgap del materiale semiconduttore (1.12 eV per Si cristallino)
    "eg": 1.12,
    # [MIGLIORAMENTO #7] Parametri di perdita di sistema
    "dc_ac_efficiency": 0.97,  # efficienza inverter DC/AC
    "mismatch_loss": 0.02,     # perdite per mismatch tra moduli (2%)
    "wiring_loss": 0.015,      # perdite per cablaggio (1.5%)
    "soiling_loss": 0.03,      # perdite per sporcizia (3%)
    # [MIGLIORAMENTO #3] Tasso di degradazione annuale (0.5%/anno)
    "annual_degradation": 0.005,
}

systems_dict = {'pv1':
                    {'tech': {"id": "pv1", "lat": 41.9027835,
                              "lon": 12.496365, "n_series": 30, "n_parallel": 10, "tilt": 30, "azimuth": 0,
                              **pv_module_params},
                     'economics': {'cap_cost': 1500, 'opex_cost': 40, 'inc_year': min(0.5*30*10*0.4*1500,96000)/10, 'inc_start_end': [1, 10],
                                   'tax_year': 0, 'other_cost': {'item1': {'unit': 30*10*0.4, 'cost_unit': 350, 'dur': [10, 10]}},
                                   'other_rev': {'item1': {'unit': 0, 'rev_unit': 0, 'dur': [0, 0]}}}, },
                'pv2':
                    {'tech': {"id": "pv1", "lat": 41.891159,
                              "lon": 12.492059, "n_series": 13, "n_parallel": 12, "tilt": 30, "azimuth": 0,
                              **pv_module_params},
                     'economics': {'cap_cost': 1500, 'opex_cost': 40, 'inc_year': min(0.5*13*12*0.4*1500,96000)/10, 'inc_start_end': [1, 10],
                                   'tax_year': 0, 'other_cost': {'item1': {'unit': 13*12*0.4, 'cost_unit': 350, 'dur': [10, 10]}},
                                   'other_rev': {'item1': {'unit': 0, 'rev_unit': 0, 'dur': [0, 0]}}}, },
                'pv3':
                    {'tech': {"id": "pv1", "lat": 41.893483,
                              "lon": 12.492477, "n_series": 11, "n_parallel": 11, "tilt": 30, "azimuth": 0,
                              **pv_module_params},
                     'economics': {'cap_cost': 1500, 'opex_cost': 40, 'inc_year':0, 'inc_start_end': [0, 0],
                                   'tax_year': 0, 'other_cost': {'item1': {'unit': 11*11*0.4, 'cost_unit': 350, 'dur': [10, 10]}},
                                   'other_rev': {'item1': {'unit': 0, 'rev_unit': 0, 'dur': [0, 0]}}}, }
                }

# [MIGLIORAMENTO BESS] Parametri comuni per tutte le batterie.
# Questi parametri erano assenti nel modello originale e sono ora configurabili.
bess_common_tech_params = {
    # [MIGLIORAMENTO #1] Efficienza di carica/scarica (round-trip = 0.95*0.95 = 90.25%)
    'eta_charge': 0.95,
    'eta_discharge': 0.95,
    # [MIGLIORAMENTO #2] Tensione minima a SOC=0 per modello V(SOC) lineare.
    # Per moduli da 25.6V nominale, V_min tipico ~20V (cella LFP ~2.5V * 8 celle)
    'v_min': 20.0,
    # [MIGLIORAMENTO #5] Tasso di autoscarica orario (~3%/mese per Li-ion)
    'self_discharge_rate_per_hour': 0.00004,
    # [MIGLIORAMENTO #9] C-rate massimo (1C = scarica completa in 1h)
    'c_rate_max': 1.0,
    # [MIGLIORAMENTO #3] Degradazione annuale della capacità (2%/anno)
    'annual_capacity_fade': 0.02,
    # [MIGLIORAMENTO #10] Vita utile della batteria (anni)
    'lifetime_years': 15,
    # [MIGLIORAMENTO #6] Soglia minima per evitare micro-cicli (1% della capacità)
    'min_energy_threshold': 0.01,
}

bess_dict = {'bess1':
                 {'tech': {"id": "bess1", 'cap_module': 2.560, 'v': 25.6, 'i_max': 100,
                           'i_min': 5, 'soc_in': 0.2, 'soc_max': 0.8, 'soc_min': 0.2, 'n_series': 2, 'n_parallel': 2,
                           **bess_common_tech_params},
                  'economics': {'cap_cost': 720, 'opex_cost': 20, 'inc_year': min(0.5*720*2.560*2*2,96000)/10, 'inc_start_end': [1, 10],
                                'tax_year': 10, 'other_cost': {'item1': {'unit': 2.560*2*2, 'cost_unit': 720, 'dur': [10, 10]}},
                                'other_rev': {'item1': {'unit': 0, 'rev_unit': 0, 'dur': [0, 0]}}}, },
             'bess2':
                 {'tech': {"id": "bess2", 'cap_module': 2.560, 'v': 25.6, 'i_max': 100,
                           'i_min': 5, 'soc_in': 0.2, 'soc_max': 1, 'soc_min': 0.2, 'n_series': 2, 'n_parallel': 2,
                           **bess_common_tech_params},
                  'economics': {'cap_cost': 720, 'opex_cost': 20, 'inc_year':min(0.5*720*2.560*2*2,96000)/10, 'inc_start_end': [1, 10],
                                'tax_year': 10, 'other_cost': {'item1': {'unit': 2.560*2*2, 'cost_unit': 720, 'dur': [10, 10]}},
                                'other_rev': {'item1': {'unit': 0, 'rev_unit': 0, 'dur': [0, 0]}}}, },
             'bess3':
                 {'tech': {"id": "bess2", 'cap_module': 2.560, 'v': 25.6, 'i_max': 100,
                           'i_min': 5, 'soc_in': 0.2, 'soc_max': 1, 'soc_min': 0.2, 'n_series': 2, 'n_parallel': 2,
                           **bess_common_tech_params},
                  'economics': {'cap_cost': 720, 'opex_cost': 20, 'inc_year': 0, 'inc_start_end': [0, 0],
                                'tax_year': 10, 'other_cost': {'item1': {'unit': 2.560*2*2, 'cost_unit': 720, 'dur': [10, 10]}},
                                'other_rev': {'item1': {'unit': 0, 'rev_unit': 0, 'dur': [0, 0]}}}, }
             }

prosumers_dict = {
    "prosumer1": {'tech': {'id': 'prosumer1', 'carriers': ['electricity'], 'users': ['user0'],
                           'systems': ['pv1'],
                           'bess': ['bess1','bess2']},
                  "economics": {
                      "tax_rate": 0.2,
                      "int_rate": 0.03, 'carriers_and_costs': {
                          'electricity': {'decay': 0.02, 'price_buy': 130, 'price_sold': 104, 'prod_degradation': 0.005}},
                      'other_capex_perc': [0]

                  }},
    "prosumer2": {'tech': {'id': 'prosumer2', 'carriers': ['electricity'], 'users': ['user1'],
                           'systems': ['pv2'],
                           'bess': []},
                  "economics": {
                      "tax_rate": 0.2,
                      "int_rate": 0.03, 'carriers_and_costs': {
                          'electricity': {'decay': 0.02, 'price_buy': 130, 'price_sold': 104, 'prod_degradation': 0.005}},
                      'other_capex_perc': [0.02]

                  }}
}

rec_dict = {'rec1': {'tech': {
    "id": "rec1",
    'carriers': ['electricity'],
    'prosumers': ['prosumer1', 'prosumer2'],
    'consumers': consumers_list,
    'rec_systems': ['pv3'], 'bess': ['bess3']},

    "economics": {
        "tax_rate": 0.2,
        "int_rate": 0.03,
        'carriers_and_costs': {'electricity': {'decay': 0.02, 'price_buy': 130, 'price_sold':104}},
        'other_capex_perc': [0.02]

    }
}}

config_data = {
    "simulation": simulation_dict,

    "users": [users_dict],

    "systems": [systems_dict]
    ,
    "bess": [bess_dict]
    ,
    "prosumers": [prosumers_dict],

    "rec": [rec_dict]
}

with open("Input/config.yaml", "w") as file:
    yaml.dump(config_data, file, sort_keys=False)

print("✅ File 'config.yaml' successfully created.")
