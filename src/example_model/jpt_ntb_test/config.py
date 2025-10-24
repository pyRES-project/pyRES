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

systems_dict = {'pv1':
                    {'tech': {"id": "pv1", "lat": 41.9027835,
                              "lon": 12.496365, "n_series": 30, "n_parallel": 10, "tilt": 30, "azimuth": 0},
                     'economics': {'cap_cost': 1500, 'opex_cost': 40, 'inc_year': min(0.5*30*10*0.4*1500,96000)/10, 'inc_start_end': [1, 10],
                                   'tax_year': 0, 'other_cost': {'item1': {'unit': 30*10*0.4, 'cost_unit': 350, 'dur': [10, 10]}},
                                   'other_rev': {'item1': {'unit': 0, 'rev_unit': 0, 'dur': [0, 0]}}}, },
                'pv2':
                    {'tech': {"id": "pv1", "lat": 41.891159,
                              "lon": 12.492059, "n_series": 13, "n_parallel": 12, "tilt": 30, "azimuth": 0},
                     'economics': {'cap_cost': 1500, 'opex_cost': 40, 'inc_year': min(0.5*13*12*0.4*1500,96000)/10, 'inc_start_end': [1, 10],
                                   'tax_year': 0, 'other_cost': {'item1': {'unit': 13*12*0.4, 'cost_unit': 350, 'dur': [10, 10]}},
                                   'other_rev': {'item1': {'unit': 0, 'rev_unit': 0, 'dur': [0, 0]}}}, },
                'pv3':
                    {'tech': {"id": "pv1", "lat": 41.893483,
                              "lon": 12.492477, "n_series": 11, "n_parallel": 11, "tilt": 30, "azimuth": 0},
                     'economics': {'cap_cost': 1500, 'opex_cost': 40, 'inc_year':0, 'inc_start_end': [0, 0],
                                   'tax_year': 0, 'other_cost': {'item1': {'unit': 11*11*0.4, 'cost_unit': 350, 'dur': [10, 10]}},
                                   'other_rev': {'item1': {'unit': 0, 'rev_unit': 0, 'dur': [0, 0]}}}, }
                }

bess_dict = {'bess1':
                 {'tech': {"id": "bess1", 'cap_module': 2.560, 'v': 25.6, 'i_max': 100,
                           'i_min': 5, 'soc_in': 0.2, 'soc_max': 0.8, 'soc_min': 0.2, 'n_series': 2, 'n_parallel': 2},
                  'economics': {'cap_cost': 720, 'opex_cost': 20, 'inc_year': min(0.5*720*2.560*2*2,96000)/10, 'inc_start_end': [1, 10],
                                'tax_year': 10, 'other_cost': {'item1': {'unit': 2.560*2*2, 'cost_unit': 720, 'dur': [10, 10]}},
                                'other_rev': {'item1': {'unit': 0, 'rev_unit': 0, 'dur': [0, 0]}}}, },
             'bess2':
                 {'tech': {"id": "bess2", 'cap_module': 2.560, 'v': 25.6, 'i_max': 100,
                           'i_min': 5, 'soc_in': 0.2, 'soc_max': 1, 'soc_min': 0.2, 'n_series': 2, 'n_parallel': 2},
                  'economics': {'cap_cost': 720, 'opex_cost': 20, 'inc_year':min(0.5*720*2.560*2*2,96000)/10, 'inc_start_end': [1, 10],
                                'tax_year': 10, 'other_cost': {'item1': {'unit': 2.560*2*2, 'cost_unit': 720, 'dur': [10, 10]}},
                                'other_rev': {'item1': {'unit': 0, 'rev_unit': 0, 'dur': [0, 0]}}}, },
             'bess3':
                 {'tech': {"id": "bess2", 'cap_module': 2.560, 'v': 25.6, 'i_max': 100,
                           'i_min': 5, 'soc_in': 0.2, 'soc_max': 1, 'soc_min': 0.2, 'n_series': 2, 'n_parallel': 2},
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
                          'electricity': {'decay': 0.02, 'price_buy': 130, 'price_sold': 104}},
                      'other_capex_perc': [0]

                  }},
    "prosumer2": {'tech': {'id': 'prosumer2', 'carriers': ['electricity'], 'users': ['user1'],
                           'systems': ['pv2'],
                           'bess': []},
                  "economics": {
                      "tax_rate": 0.2,
                      "int_rate": 0.03, 'carriers_and_costs': {
                          'electricity': {'decay': 0.02, 'price_buy': 130, 'price_sold': 104}},
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
