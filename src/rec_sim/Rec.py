"""
Created on June 7 08:00:00 2025

@author: isabella pizzuti
"""

import numpy as np
from src.rec_sim.Economics import Economics
from src.rec_sim.Controller import Controller


class Rec:
    def __init__(self, id, carriers, prosumers, consumers, rec_systems=[],rec_bess=[]):
        """
        simulates a REC which includes prosumers, consumers and power production system of the REC

        :param id: str --> identification code  e.g: 'rec1'
        :param carriers: list of str--> list of carriers e.g: ['electricity','heat']
        :param prosumers: list of obj by Prosumer --> list of prosumers
        :param consumers: list of obj by Consumer --> list of consumers
        :param rec_systems: list of obj by System --> list of production systems
        :param rec_bess: list of obj by Bess --> list of batteries
        """

        self.id = id
        self.carriers = carriers
        self.prosumers = prosumers
        self.consumers = consumers
        self.rec_systems = rec_systems
        self.rec_bess = rec_bess
        self.en_perf_evolution = {}
        self.ec_perf={}



    def compute_members(self):
        """
        :return:
            n_members: int
            n_prosumers: int
            n_consumers: int
        """
        if self.prosumers:
            n_prosumers = len(self.prosumers)
        else:
            n_prosumers = 0
        if self.consumers:
            n_consumers = len(self.consumers)
        else:
            n_consumers = 0

        n_members = n_prosumers + n_consumers

        return n_members, n_prosumers, n_consumers

    def energy_performance(self, time):
        """

        :param time: float--> 1 if hourly analysis, 0.25 if quarterly analysis
        :return: en_perf_evolution: dict--> contains for each carrier:
        - prod : DataSeries or array--> (kW) energy production from all prosumers and REC systems
        - prod_net : DataSeries or array--> (kW) net energy production  defined as the total production from all prosumers and REC systems minus self-consumption from all prosumers.
        - prod_rec : DataSeries or array--> (kW) energy production from REC systems only
        - dem : DataSeries or array--> (kW) energy demand  from all consumers and prosumers
        - dem_net : DataSeries or array--> (kW) net energy demand (kW) defined as the total demand from all consumers and prosumers minus self-consumption from all prosumers.
        - shared : DataSeries or array--> (kW)shared energy (kW) defined as the minimum between net energy production and net energy demand in each time step. If BESS is present, shared includes the energy stored in the BESS.
        - surplus_prosumer : DataSeries or array--> (kW) surplus production from all prosumers
        - selfcons_prosumer : DataSeries or array--> (kW)self-consumption from all prosumers
        - unmet_prosumers	: DataSeries or array--> (kW) deficit from all prosumers and consumers
        - surplus: DataSeries or array--> (kW) surplus production from all prosumers and REC systems
        - unmet	: DataSeries or array--> (kW) deficit from all prosumers and consumers
        - stored : DataSeries or array--> (kW) energy stored in BESS managed by REC
        - supply : DataSeries or array--> (kW) energy supplied by BESS managed by REC
        - power	: DataSeries or array--> (kW) energy exchanged with BESS managed by REC
        - soc : DataSeries or array --> (%) state of charge of BESS managed by REC


        """
        for carrier in  self.carriers:

            d_consumers=0
            d_prosumers=0
            p_prosumers=0
            p_rec = 0
            surplus_prosumers=0
            selfcons_prosumers=0
            deficit_prosumers=0



            for prosumer in self.prosumers:
                if carrier in prosumer.carriers:
                    p_prosumers+= prosumer.en_perf_evolution[carrier]['prod']
                    d_prosumers+= prosumer.en_perf_evolution[carrier]['dem']
                    surplus_prosumers+= prosumer.en_perf_evolution[carrier]['surplus']
                    selfcons_prosumers+= prosumer.en_perf_evolution[carrier]['self_cons']
                    deficit_prosumers+= prosumer.en_perf_evolution[carrier]['unmet']

            for consumer in self.consumers:
                if carrier in consumer.dem:
                    d_consumers += consumer.en_perf_evolution[carrier]

            for plant in self.rec_systems:
                if carrier in plant.carriers:
                    p_rec= plant.en_perf_evolution[carrier]['prod']


            d_tot = d_prosumers + d_consumers
            d_net = deficit_prosumers+d_consumers
            p_tot = p_prosumers+p_rec
            p_net = surplus_prosumers+p_rec


            if np.isscalar(p_rec) and p_rec == 0:
                p_rec = np.zeros(len(p_tot))
            shared = np.zeros(len(p_tot))
            surplus_rec =np.zeros(len(p_tot))
            deficit_rec = np.zeros(len(p_tot))

            for dt in range(len(p_tot)):
                p = p_net[dt]
                d = d_net[dt]
                if p >= d:
                    surplus_rec[dt] = p - d
                    deficit_rec[dt] = 0
                    shared[dt] = d
                else:
                    surplus_rec[dt] = 0
                    deficit_rec[dt] = d - p
                    shared[dt] = p

            self.en_perf_evolution[carrier] = {}
            self.en_perf_evolution[carrier]['prod'] = p_tot
            self.en_perf_evolution[carrier]['prod_net'] = p_net
            self.en_perf_evolution[carrier]['prod_rec'] = p_rec
            self.en_perf_evolution[carrier]['dem'] = d_tot
            self.en_perf_evolution[carrier]['dem_net'] = d_net
            self.en_perf_evolution[carrier]['shared'] = shared
            self.en_perf_evolution[carrier]['surplus_prosumers'] = surplus_prosumers
            self.en_perf_evolution[carrier]['selfcons_prosumers'] = selfcons_prosumers
            self.en_perf_evolution[carrier]['unmet_prosumers'] = deficit_prosumers
            self.en_perf_evolution[carrier]['surplus'] = surplus_rec
            self.en_perf_evolution[carrier]['unmet'] = deficit_rec

            if carrier=='electricity':
                if self.rec_bess:

                    bess = self.rec_bess
                    controller = Controller(bess=bess)

                    stored, supply, power, surplus_rec, deficit_rec, soc = controller.energy_performance(
                        production=p_net, demand=d_net, time=time)
                    self.en_perf_evolution[carrier]['stored'] = stored
                    self.en_perf_evolution[carrier]['supply'] = supply
                    self.en_perf_evolution[carrier]['power'] = power
                    self.en_perf_evolution[carrier]['soc'] = soc
                    self.en_perf_evolution[carrier]['shared'] = shared + stored
                    self.en_perf_evolution[carrier]['surplus'] = surplus_rec
                    self.en_perf_evolution[carrier]['unmet'] = deficit_rec


        return self.en_perf_evolution



    def economic_performance(self, time_horizon, tax_rate, int_rate, other_capex_perc, annual_en_flows_and_price):
        """

        :param time_horizon: int--> investment time horizon (year)
        :param tax_rate: float--> tax on revenues from sale e.g 0.2
        :param int_rate: float-->interest rate for calculating NPV e.g 0.03
        :param other_capex_perc: float--> list of other capex as percentage of total capex e.g [0.2,0.5]
        :param annual_en_flows_and_price: dict--> e.g. annnual_en_flows={'electricity':{'sold':100,'self_cons':200,'purchased':10,'price_sold':2,'price_buy':3,'decay':0.02}}
        :return: ec_perf: dict : e.g. ec_perf={'NPV':value,'pbp':value,'capex':value,'rev_from_sale':r1,'rev_savings':r2,'rev_incentives':r3,'rev_others':r4,'cost_resources':c1,'cost_opex':c2,'cost_taxes':c3,'cost_taxes_on_sale':c4,'cost_others':c5}
        """

        calculator = Economics(components=self.rec_systems+self.rec_bess, annual_en_flows_and_prices=annual_en_flows_and_price)
        ec_perf = calculator.compute_cashflow(time_horizon=time_horizon, tax_rate=tax_rate, int_rate=int_rate, other_capex_perc=other_capex_perc)
        self.ec_perf = ec_perf
        return ec_perf
