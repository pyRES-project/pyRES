"""
Created on June 7 08:00:00 2025

@author: isabella pizzuti
"""

import numpy as np
from src.rec_sim.Economics import Economics
from src.rec_sim.Controller import Controller



class Prosumer:
    def __init__(self, id, carriers, systems, users, bess=[]):
        """
        simulates the demand-production coupling, integrating one or more energy consumers with one or more production systems. Electric batteries are optional.

        :param id: str --> identification code  example: 'prosumer1'
        :param carriers : list of str --> e.g ['electricity','heat']
        :param systems: list of obj by System --> list of production power system of prosumer  example: [pv1,wt2]
        :param users: list of obj by Consumer --> list of consumers physically connected to plants example:[consumer1,consumer2]
        :param bess: list of obj by Bess--> list of battery storage  example: [battery1,battery2]
        """

        self.id = id
        self.carriers = carriers
        self.users = users
        self.systems = systems
        self.bess = bess


        self.en_perf_evolution = {}
        self.ec_perf = {}


    def energy_performance(self, time):
        """

        :param time: float--> 1 if hourly analysis, 0.25 if quarterly analysis
        :return: en_perf_evolution: dict--> contains for each carrier:
            prod: DataSeries or array--> (kW) energy production
            dem:  DataSeries or array-->(kW) energy demand
            self_cons: DataSeries or array--> (kW) self-consumption defined as the minimum between production and demand in each time step. If BESS is present, self-consumption includes the energy stored in the BESS.
            surplus: DataSeries or array--> (kW) surplus production defined as the production exceeding the demand in each time step. If BESS is present, it takes in account the energy stored in the BESS.
            unmet: DataSeries or array--> (kW) defined as the demand exceeding the production in each time step. If BESS is present, it takes in account the energy supplied by the BESS.
            self_cons_without_bess: DataSeries or array--> (kW) self-consumption defined as the minimum between production and demand in each time step.
            surplus_without_bess: DataSeries or array--> (kW) surplus production defined as the production exceeding the demand in each time step.
            unmet_without_bess: DataSeries or array--> (kW) defined as the demand exceeding the production in each time step.
            stored: DataSeries or array--> (kW) energy stored in BESS
            supply: DataSeries or array--> (kW) energy supplied by BESS
            power: DataSeries or array--> (kW) energy exchanged with BESS
            soc: DataSeries or array --> (%) state of charge of BESS

        """

        for carrier in self.carriers:
            d_tot = 0
            for consumer in self.users:
                if carrier in consumer.dem.keys():
                    d_tot += consumer.en_perf_evolution[carrier]
                else:
                    d_tot += 0

            p_tot = 0
            for system in self.systems:
                if carrier in system.carriers:
                    p_tot += system.en_perf_evolution[carrier]['prod']
                else:
                    p_tot += 0

            self_cons = np.zeros(len(d_tot))
            surplus = np.zeros(len(d_tot))
            unmet = np.zeros(len(d_tot))
            for dt in range(len(d_tot)):
                p = p_tot[dt]
                d = d_tot[dt]
                if p >= d:
                    surplus[dt] = p - d
                    unmet[dt] = 0
                    self_cons[dt] = d

                else:
                    surplus[dt] = 0
                    unmet[dt] = d - p
                    self_cons[dt] = p

            self.en_perf_evolution[carrier] = {}
            self.en_perf_evolution[carrier]['prod'] = p_tot
            self.en_perf_evolution[carrier]['dem'] = d_tot
            self.en_perf_evolution[carrier]['self_cons'] = self_cons
            self.en_perf_evolution[carrier]['surplus'] = surplus
            self.en_perf_evolution[carrier]['unmet'] = unmet

            if carrier=='electricity':
                if self.bess:
                    self.en_perf_evolution[carrier] = {}
                    self.en_perf_evolution[carrier]['prod'] = p_tot
                    self.en_perf_evolution[carrier]['dem'] = d_tot
                    self.en_perf_evolution[carrier]['self_cons_without_bess'] = self_cons
                    self.en_perf_evolution[carrier]['surplus_without_bess'] = surplus
                    self.en_perf_evolution[carrier]['unmet_without_bess'] = unmet

                    bess = self.bess
                    controller = Controller(bess=bess)

                    stored, supply, power, surplus, deficit, soc = controller.energy_performance(
                        production=p_tot, demand=d_tot, time=time)
                    self.en_perf_evolution[carrier]['stored'] = stored
                    self.en_perf_evolution[carrier]['supply'] = supply
                    self.en_perf_evolution[carrier]['power'] = power
                    self.en_perf_evolution[carrier]['soc'] = soc
                    self.en_perf_evolution[carrier]['self_cons'] = self_cons + stored
                    self.en_perf_evolution[carrier]['surplus'] = surplus
                    self.en_perf_evolution[carrier]['unmet'] = deficit

        return self.en_perf_evolution

    def economic_performance(self, time_horizon, tax_rate, int_rate, other_capex_perc, annual_en_flows_and_price):
        """

        :param time_horizon: int--> investment time horizon (year)
        :param tax_rate: float--> tax on revenues from sale e.g 0.2
        :param int_rate: float--> interest rate for calculating NPV e.g 0.03
        :param other_capex_perc: float--> list of other capex as percentage of total capex e.g [0.2,0.5]
        :param annual_en_flows_and_price: dict--> e.g. annnual_en_flows={'electricity':{'sold':100,'self_cons':200,'purchased':10,'price_sold':2,'price_buy':3,'decay':0.02}}
        :return: ec_perf: dict : e.g. ec_perf={'NPV':value,'pbp':value,'capex':value,'rev_from_sale':r1,'rev_savings':r2,'rev_incentives':r3,'rev_others':r4,'cost_resources':c1,'cost_opex':c2,'cost_taxes':c3,'cost_taxes_on_sale':c4,'cost_others':c5}
        """

        calculator = Economics(components=self.systems+self.bess, annual_en_flows_and_prices=annual_en_flows_and_price)
        ec_perf = calculator.compute_cashflow(time_horizon=time_horizon, tax_rate=tax_rate, int_rate=int_rate, other_capex_perc=other_capex_perc)
        self.ec_perf = ec_perf
        return ec_perf
