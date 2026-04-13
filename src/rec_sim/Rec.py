"""
Created on June 7 08:00:00 2025

@author: isabella pizzuti
"""

import numpy as np
from src.rec_sim.Economics import Economics
from src.rec_sim.Environmentals import Environmentals
from src.rec_sim.Controller import Controller


class Rec:
    def __init__(self, id, carriers, prosumers, consumers, rec_systems=[],rec_bess=[], rec_users=[]):
        """
        simulates a REC which includes prosumers, consumers and power production system of the REC

        :param id: str --> identification code  e.g: 'rec1'
        :param carriers: list of str--> list of carriers e.g: ['electricity','heat']
        :param prosumers: list of obj by Prosumer --> list of prosumers
        :param consumers: list of obj by Consumer --> list of consumers
        :param rec_systems: list of obj by System --> list of production systems
        :param rec_bess: list of obj by Bess --> list of batteries
        :param rec_users: list of obj by Consumer --> loads directly connected to the REC (optional)
        """

        self.id = id
        self.carriers = carriers
        self.prosumers = prosumers
        self.consumers = consumers
        self.rec_systems = rec_systems
        self.rec_bess = rec_bess
        self.rec_users = rec_users
        self.en_perf_evolution = {}
        self.ec_perf = {}
        self.env_perf = {}



    def compute_members(self):
        """
        :return:
            n_members: int
            n_prosumers: int
            n_consumers: int
        """
        n_prosumers = len(self.prosumers) if self.prosumers else 0
        n_consumers = len(self.consumers) if self.consumers else 0
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
                    p_rec += plant.en_perf_evolution[carrier]['prod']

            # [FIX #1] Conversione di p_rec ad array PRIMA dell'uso nelle somme.
            # Nel codice originale, quando rec_systems era vuoto p_rec restava scalare (0)
            # e veniva usato nelle somme per p_tot e p_net prima di essere convertito,
            # funzionando per caso grazie al broadcasting di NumPy ma in modo fragile.
            # Ora determiniamo la lunghezza della serie temporale e convertiamo subito.
            ref_len = None
            if not np.isscalar(p_prosumers):
                ref_len = len(p_prosumers)
            elif not np.isscalar(d_consumers):
                ref_len = len(d_consumers)
            elif not np.isscalar(d_prosumers):
                ref_len = len(d_prosumers)

            if np.isscalar(p_rec) and ref_len is not None:
                p_rec = np.zeros(ref_len)

            d_tot = d_prosumers + d_consumers
            d_net = deficit_prosumers + d_consumers
            p_tot = p_prosumers + p_rec
            p_net = surplus_prosumers + p_rec

            # [FIX #6] Vettorizzazione: il loop Python esplicito è sostituito
            # da operazioni NumPy vettoriali per il calcolo di shared/surplus/deficit.
            p_net = np.asarray(p_net, dtype=float)
            d_net = np.asarray(d_net, dtype=float)
            shared = np.minimum(p_net, d_net)
            surplus_rec = np.maximum(0, p_net - d_net)
            deficit_rec = np.maximum(0, d_net - p_net)

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

            # [MIGLIORAMENTO #8] BESS multi-carrier
            if self.rec_bess:
                carrier_bess = [b for b in self.rec_bess if carrier in b.carriers]
                if carrier_bess:
                    controller = Controller(bess=carrier_bess)

                    stored, supply, power, surplus_rec, deficit_rec, soc = controller.energy_performance(
                        production=p_net, demand=d_net, time=time)
                    self.en_perf_evolution[carrier]['stored'] = stored
                    self.en_perf_evolution[carrier]['supply'] = supply
                    self.en_perf_evolution[carrier]['power'] = power
                    self.en_perf_evolution[carrier]['soc'] = soc
                    self.en_perf_evolution[carrier]['shared'] = shared + stored
                    self.en_perf_evolution[carrier]['surplus'] = surplus_rec
                    self.en_perf_evolution[carrier]['unmet'] = deficit_rec


        # Compute annual values (MWh) for all timeseries in each carrier
        for carrier in self.carriers:
            ep = self.en_perf_evolution[carrier]
            annual = {}
            for key, val in ep.items():
                if key == 'soc':
                    continue
                arr = np.asarray(val, dtype=float)
                annual[key] = float(np.sum(arr)) * time / 1000  # kW * h → MWh
            # SC and SS rates (selfcons_prosumers + shared vs prod and dem)
            sc_total = annual.get('selfcons_prosumers', 0) + annual.get('shared', 0)
            prod = annual.get('prod', 0)
            dem = annual.get('dem', 0)
            annual['sc_total'] = sc_total
            annual['sc_rate'] = (sc_total / prod * 100) if prod > 0 else 0
            annual['ss_rate'] = (sc_total / dem * 100) if dem > 0 else 0
            ep['annual'] = annual

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

    def environmental_performance(self, time_horizon, time_step):
        """

        :param time_horizon: int--> investment time horizon (year)
        :param time_step: float--> 1 if hourly analysis, 0.25 if quarterly analysis
        :return: env_perf: dict with CO2, GWP, fuel savings and CRM indicators
        """

        # Only REC-owned assets (same logic as economic_performance)
        annual_prod_kwh = 0
        for carrier in self.carriers:
            ep = self.en_perf_evolution.get(carrier, {})
            annual_prod_kwh += float(np.sum(ep.get('prod_rec', 0))) * time_step

        calculator = Environmentals(
            components=self.rec_systems + self.rec_bess,
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
        n_members, _, _ = self.compute_members()

        result = {'Entity': f"{self.id} ({n_members} members)"}

        # Energy annual (MWh/y)
        for key, label in [
            ('prod', 'Production [MWh/y]'),
            ('prod_rec', 'Production REC [MWh/y]'),
            ('dem', 'Demand [MWh/y]'),
            ('shared', 'Shared energy [MWh/y]'),
            ('selfcons_prosumers', 'SC prosumers [MWh/y]'),
            ('surplus', 'Surplus [MWh/y]'),
            ('unmet', 'Purchased [MWh/y]'),
        ]:
            if key in an:
                result[label] = round(an[key], 1)

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
