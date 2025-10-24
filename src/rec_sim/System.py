"""
Created on June 7 08:00:00 2025

@author: isabella pizzuti
"""


class System():
    def __init__(self, id, carriers, cap, cap_cost, opex, opex_cost, inc_year, inc_start_end, tax_year,
                 other_cost={'item1': {'unit': 0, 'cost_unit': 0, 'dur': [0, 0]}},
                 other_rev={'item1': {'unit': 0, 'rev_unit': 0, 'dur': [0, 0]}}):

        """

        :param id: str--> id code
        :param carriers: list of str--> list of carriers e.g. ['electricity']
        :param cap: float --> capacity e.g kW, m3, kg
        :param cap_cost: float --> €/unit initial cost
        :param opex: float --> capacity e.g kW, m3, kg, Represents the portion of the total capacity associated with maintenance-related operating costs.
        :param opex_cost: float --> €/unit operating cost
        :param inc_year: float --> €/year incentives on the system
        :param inc_start_end: list --> start and end date in year e.g. [1,6], means that incentives (inc_year) are active from month 1 to month 6
        :param tax_year: float --> €/year taxes on the system
        :param other_cost: dic --> e.g. {'item1': {'unit': 0, 'cost_unit': 0, 'dur': [0, 0]}} is a dictionary used to define additional costs. Each entry (e.g. 'item1') includes:
            - unit: the unit of measurement for the cost (e.g. kWh, kg, etc.)
            - cost_unit: the unit cost value (e.g. €/kWh, €/kg, etc.)
            - dur: a list representing the start and end of the cost duration
            - e.g. {'item1': {'unit':10, 'cost_unit': 720, 'dur': [8, 10]}} simulates a cost of 720 €/unit per 10 unit, 7200 € will be added from year 8 to year 10.
        :param other_rev: dic--> e.g {'item1': {'unit': 0, 'cost_unit': 0, 'dur': [0, 0]}} is a dictionary used to define additional revenues. Each entry (e.g., 'item1') includes:
            - unit: the unit of measurement for the revenue (e.g. kWh, kg, etc.)
            - rev_unit: the unit revenue value (e.g €/kWh, €/kg, etc.)
            - dur: a list representing the start and end of the revenue duration.
            - e.g. {'item1': {'unit': 50, 'rev_unit': 4, 'dur': [1, 2]}} from year1 to year2 an additional revenue of 200 € will be applied.
        """
        self.id = id
        self.carriers = carriers
        self.cap_cost_unit = cap_cost
        self.cap = cap
        self.opex_cost = opex_cost
        self.opex = opex
        self.inc_year = inc_year
        self.inc_start_end = inc_start_end
        self.tax_year = tax_year
        self.other_cost = other_cost
        self.other_rev = other_rev
        self.en_perf_evolution = {}


