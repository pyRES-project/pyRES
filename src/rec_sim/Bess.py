"""
Created on June 5 08:00:00 2025

@author: isabella pizzuti
"""

from src.rec_sim.System import System

class Bess(System):
    def __init__(self, id,  cap_module, cap_cost, opex_cost, inc_year, inc_start_end, tax_year,
                          v, i_max,
                 i_min, soc_in, soc_max, soc_min, n_series, n_parallel,carriers=['electricity'],other_cost={'item1': {'unit': 0, 'cost_unit': 0, 'dur': [0, 0]}},
                 other_rev={'item1': {'unit': 0, 'rev_unit': 0, 'dur': [0, 0]}}):

        """

        :param id:  str--> id code
        :param cap_module: float --> single module capacity  kWh
        :param cap_cost: float --> €/kWh initial cost
        :param opex: float --> single module capacity for O&M  kWh
        :param opex_cost: float --> €/kWh operating cost
        :param inc_year: float --> €/year incentives on the system
        :param inc_start_end:  list --> start and end date in year e.g. [1,6]
        :param tax_year: float --> €/year taxes on the system
        :param other_cost: dict--> e.g. {'item1': {'unit': 0, 'cost_unit': 0, 'dur': [0, 0]}}
        :param other_rev:  dict--> e.g. {'item1': {'unit': 0, 'cost_unit': 0, 'dur': [0, 0]}}
        :param v:float --> rated voltage[V]
        :param i_max:float -->max. current per cell charging and discharge [A]
        :param i_min:float -->min. current per cell charging and discharge [A]
        :param soc_in:float -->initial state of charge
        :param soc_max:float -->max. state of charge
        :param soc_min:float -->min. state of charge
        :param n_series:int --> modules connected in series
        :param n_parallel:int -->modules connected in parallel
        """

        self.id = id
        self.soc_in = soc_in
        self.soc_max = soc_max
        self.soc_min = soc_min
        self.n_series = n_series
        self.n_parallel = n_parallel
        self.cap = cap_module * self.n_parallel * self.n_series
        self.opex = cap_module * self.n_parallel * self.n_series
        self.i_max = i_max * self.n_parallel
        self.i_min = i_min * self.n_parallel
        self.v = v * self.n_series
        self.en_perf_evolution = {}



        super().__init__(id=id,cap=self.cap, carriers=carriers, cap_cost=cap_cost, opex=self.opex, opex_cost=opex_cost,
                         inc_year=inc_year, inc_start_end=inc_start_end, tax_year=tax_year,
                         other_cost=other_cost,
                         other_rev=other_rev )

    def energy_performance(self, power_in, time):
        """
        :param power_in: float--> input power to battery (kW)
        :param time: float--> 1 if hourly analysis, 0.25 if quarterly analysis
        :return:
                power_in--> power sent to the bess (+) charge (-) discharge (kW)
                soc: float--> new state of charge
                supply: float--> power supplied by the battery (kW)
                stored: float--> power stored into the battery (kW)
                power: float-->  power exchange with the battery (+) charge (-) discharge (kW)
                surplus:float--> surplus production (kW) defined as the production exceeding the battery capacity in each time step.
                deficit: float--> deficit (kW) defined as the demand exceeding the battery capacity in each time step.
                current: float--> current (A)
                mode: int --> operation mode
        """


        soc = self.soc_in
        energy_in = power_in * time
        energy_min = self.v * time * self.i_min / 1000
        energy_max = self.v * time * self.i_max/ 1000
        if energy_in > 0:
            avaliability = self.cap * (self.soc_max - soc)
            if soc < self.soc_max:
                if energy_in >= avaliability:
                    charge = avaliability
                    if charge > energy_max:
                        charge = energy_max
                        current = self.i_max
                        mode = 1
                    elif charge < energy_min:
                        charge = 0
                        current = 0
                        mode = 2
                    else:
                        charge = avaliability
                        current = charge*1000/ (time * self.v)
                        mode = 3

                    surplus = energy_in - charge
                    deficit = 0
                    supply = 0
                    battery = charge
                    stored = battery

                else:
                    charge = energy_in
                    if charge > energy_max:
                        charge = energy_max
                        current = self.i_max
                        mode = 4
                    elif charge < energy_min:
                        charge = 0
                        current = 0
                        mode = 5
                    else:
                        charge = energy_in
                        current = charge*1000 / (time * self.v)
                        mode = 6


                    surplus = energy_in - charge
                    deficit = 0
                    battery = charge
                    supply = 0
                    stored = battery
            else:
                surplus = energy_in
                deficit = 0
                battery = 0
                stored = battery
                supply = 0
                current = 0
                mode = 7

        else:
            energy_out = -energy_in
            avaliability = self.cap * (soc - self.soc_min)
            if soc >= self.soc_min:
                if energy_out >= avaliability:
                    discharge = avaliability
                    if discharge > energy_max:
                        discharge = energy_max
                        current = self.i_max
                        mode = 8
                    elif discharge < energy_min:
                        discharge = 0
                        current = 0
                        mode = 9
                    else:
                        discharge = avaliability
                        current = discharge*1000/ (time * self.v)
                        mode = 10

                    surplus = 0
                    deficit = energy_out - discharge
                    battery = -discharge
                    stored = 0
                    supply = -battery

                else:
                    discharge = energy_out
                    if discharge > energy_max:
                        discharge = energy_max
                        current = self.i_max
                        mode = 11
                    elif discharge < energy_min:
                        discharge = 0
                        current = 0
                        mode = 12
                    else:
                        discharge = energy_out
                        current = discharge*1000 / (time * self.v)
                        mode = 13

                    surplus = 0
                    deficit = energy_out - discharge
                    battery = -discharge
                    stored = 0
                    supply = -battery

            else:
                surplus = 0
                deficit = energy_out
                battery = 0
                stored = 0
                supply = 0
                current = 0
                mode = 14

        soc = (self.cap * soc + battery) / self.cap
        power = battery / time
        surplus = surplus / time
        deficit = deficit / time
        stored = stored / time
        supply = supply / time



        return power_in,soc, stored, supply, power, surplus, deficit, current,mode
