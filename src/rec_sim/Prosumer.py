"""
Created on June 7 08:00:00 2025

@author: isabella pizzuti
"""

import numpy as np
from src.rec_sim.Economics import Economics
from src.rec_sim.Environmentals import Environmentals
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
        self.env_perf = {}


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

            # [FIX #5] Vettorizzazione: il loop Python esplicito su 35.040 timestep
            # è sostituito da operazioni NumPy vettoriali, ~100x più veloci.
            p_tot = np.asarray(p_tot, dtype=float)
            d_tot = np.asarray(d_tot, dtype=float)
            self_cons = np.minimum(p_tot, d_tot)
            surplus = np.maximum(0, p_tot - d_tot)
            unmet = np.maximum(0, d_tot - p_tot)

            self.en_perf_evolution[carrier] = {}
            self.en_perf_evolution[carrier]['prod'] = p_tot
            self.en_perf_evolution[carrier]['dem'] = d_tot
            self.en_perf_evolution[carrier]['self_cons'] = self_cons
            self.en_perf_evolution[carrier]['surplus'] = surplus
            self.en_perf_evolution[carrier]['unmet'] = unmet

            # [MIGLIORAMENTO #8] BESS multi-carrier
            if self.bess:
                carrier_bess = [b for b in self.bess if carrier in b.carriers]
                if carrier_bess:
                    self.en_perf_evolution[carrier] = {}
                    self.en_perf_evolution[carrier]['prod'] = p_tot
                    self.en_perf_evolution[carrier]['dem'] = d_tot
                    self.en_perf_evolution[carrier]['self_cons_without_bess'] = self_cons
                    self.en_perf_evolution[carrier]['surplus_without_bess'] = surplus
                    self.en_perf_evolution[carrier]['unmet_without_bess'] = unmet

                    controller = Controller(bess=carrier_bess)

                    stored, supply, power, surplus, deficit, soc = controller.energy_performance(
                        production=p_tot, demand=d_tot, time=time)
                    self.en_perf_evolution[carrier]['stored'] = stored
                    self.en_perf_evolution[carrier]['supply'] = supply
                    self.en_perf_evolution[carrier]['power'] = power
                    self.en_perf_evolution[carrier]['soc'] = soc
                    self.en_perf_evolution[carrier]['self_cons'] = self_cons + stored
                    self.en_perf_evolution[carrier]['surplus'] = surplus
                    self.en_perf_evolution[carrier]['unmet'] = deficit

        # Compute annual values (MWh) for all timeseries in each carrier
        for carrier in self.carriers:
            ep = self.en_perf_evolution[carrier]
            annual = {}
            for key, val in ep.items():
                if key == 'soc':
                    continue
                arr = np.asarray(val, dtype=float)
                annual[key] = float(np.sum(arr)) * time / 1000  # kW * h → MWh
            # SC and SS rates
            prod = annual.get('prod', 0)
            dem = annual.get('dem', 0)
            sc = annual.get('self_cons', 0)
            annual['sc_rate'] = (sc / prod * 100) if prod > 0 else 0
            annual['ss_rate'] = (sc / dem * 100) if dem > 0 else 0
            ep['annual'] = annual

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

    def environmental_performance(self, time_horizon, time_step):
        """

        :param time_horizon: int--> investment time horizon (year)
        :param time_step: float--> 1 if hourly analysis, 0.25 if quarterly analysis
        :return: env_perf: dict with CO2, GWP, fuel savings and CRM indicators
        """

        annual_prod_kwh = 0
        for carrier in self.carriers:
            ep = self.en_perf_evolution.get(carrier, {})
            annual_prod_kwh += float(np.sum(ep.get('prod', 0))) * time_step

        calculator = Environmentals(
            components=self.systems + self.bess,
            annual_prod_kwh=annual_prod_kwh,
            time_horizon=time_horizon
        )
        env_perf = calculator.compute_environmental()
        self.env_perf = env_perf
        return env_perf

    def summary(self):
        """
        Return aggregated annual KPIs (energy + economic + environmental).
        Reads pre-computed values from en_perf_evolution['annual'],
        ec_perf and env_perf — no recomputation needed.

        :return: dict with summary indicators
        """
        an = self.en_perf_evolution.get('electricity', {}).get('annual', {})
        ec = self.ec_perf or {}
        ev = self.env_perf or {}

        result = {'Entity': self.id}

        # Energy annual (MWh/y)
        for key, label in [
            ('prod', 'Production [MWh/y]'),
            ('dem', 'Demand [MWh/y]'),
            ('self_cons', 'Self-consumption [MWh/y]'),
            ('surplus', 'Surplus [MWh/y]'),
            ('unmet', 'Purchased [MWh/y]'),
        ]:
            result[label] = round(an.get(key, 0), 1)

        if 'self_cons_without_bess' in an:
            result['Self-cons. w/o BESS [MWh/y]'] = round(an['self_cons_without_bess'], 1)
            result['Surplus w/o BESS [MWh/y]'] = round(an.get('surplus_without_bess', 0), 1)
            result['Purchased w/o BESS [MWh/y]'] = round(an.get('unmet_without_bess', 0), 1)

        result['SC rate [%]'] = round(an.get('sc_rate', 0), 1)
        result['SS rate [%]'] = round(an.get('ss_rate', 0), 1)

        # Economic
        for key, label in [
            ('capex', 'CAPEX [EUR]'), ('NPV', 'NPV [EUR]'), ('pbp', 'PBP [y]'),
        ]:
            result[label] = round(ec.get(key, 0), 1 if key == 'pbp' else 0)

        # Environmental
        for key, label in [
            ('total_pv_kwp', 'PV [kWp]'), ('total_bess_kwh', 'BESS [kWh]'),
            ('co2_avoided_annual_t', 'CO2 avoided [tCO2/y]'),
            ('co2_avoided_lifetime_t', 'CO2 avoided lifetime [tCO2]'),
            ('gwp_embodied_t', 'GWP embodied [tCO2-eq]'),
            ('gwp_net_t', 'GWP net [tCO2-eq]'),
            ('co2_payback_years', 'CO2 payback [y]'),
            ('lifecycle_ef', 'Lifecycle EF [gCO2/kWh]'),
            ('crm_pv_total_kg', 'CRM PV [kg]'),
            ('crm_bess_total_kg', 'CRM BESS [kg]'),
            ('crm_total_kg', 'CRM total [kg]'),
        ]:
            val = ev.get(key, 0)
            result[label] = round(val, 2 if 'co2_avoided' in key else 1)

        return result
