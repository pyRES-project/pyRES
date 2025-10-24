"""
Created on June 7 08:00:00 2025

@author: isabella pizzuti
"""


import numpy as np
import numpy_financial as npf

class Economics:
    def __init__(self, components, annual_en_flows_and_prices):
        """
        :param components: list of objects by System or Bess
        :param annual_en_flows_and_prices: dict : e.g. annual_en_flows_and_prices={'electricity':{'sold':100,'self_cons':200,'purchased':10,'price_sold':2,'price_buy':3,'decay':0.02}}
        """

        self.components = components
        self.annual_en_flows_and_prices = annual_en_flows_and_prices

    def compute_cashflow(self,time_horizon,tax_rate ,int_rate, other_capex_perc=[0]):
        """

        :param time_horizon: int : investment time horizon (year)
        :param tax_rate: float: tax on revenues from sale e.g 0.2
        :param int_rate: float: interest rate for calculating NPV e.g 0.03
        :param other_capex_perc: list of other capex as percentage of total capex e.g [0.2,0.5]
        :return: ec_perf: dict : e.g. ec_perf={'NPV':value,'pbp':value,'capex':value,'rev_from_sale':r1,'rev_savings':r2,'rev_incentives':r3,'rev_others':r4,'cost_resources':c1,'cost_opex':c2,'cost_taxes':c3,'cost_taxes_on_sale':c4,'cost_others':c5}
        """

        outflow = np.zeros(time_horizon + 1)
        inflow = np.zeros(time_horizon + 1)
        cashflow = np.zeros(time_horizon + 1)
        cashflow_cum = np.zeros(time_horizon + 1)
        r1 = np.zeros(time_horizon + 1)
        r2 = np.zeros(time_horizon + 1)
        r3 = np.zeros(time_horizon + 1)
        r4 = np.zeros(time_horizon + 1)
        r5 = np.zeros(time_horizon + 1)
        c1 = np.zeros(time_horizon + 1)
        c2 = np.zeros(time_horizon + 1)
        c3 = np.zeros(time_horizon + 1)
        c4 = np.zeros(time_horizon + 1)
        c5 = np.zeros(time_horizon + 1)
        outflow0 = 0
        opex_cost = 0
        tax = 0
        for component in self.components:
            outflow0+=component.cap_cost_unit*component.cap
            opex_cost+= component.opex_cost * component.opex
            tax+=component.tax_year


        total_percentage = sum(other_capex_perc)
        outflow0 = outflow0 / (1 - total_percentage)


        investment_cost = outflow0
        opex_cost = 0
        tax = 0

        outflow[0] = investment_cost
        inflow[0] = 0
        cashflow[0] = inflow[0] - outflow[0]

        r1[0] = 0
        r2[0] = 0
        r3[0] = 0
        r5[0] = 0
        c1[0] = 0
        c2[0] = 0
        c3[0] = 0
        c4[0] = 0
        c5[0] = 0


        for year in range(1, time_horizon + 1):
            r1_i = 0
            r2_i = 0
            r3_i = 0
            r4_i = 0
            c1_i = 0
            c2_i = opex_cost
            c3_i = tax
            c5_i = 0
            for key in self.annual_en_flows_and_prices:
                r1_i += self.annual_en_flows_and_prices[key]['sold'] * self.annual_en_flows_and_prices[key]['price_sold'] * (
                        1 - self.annual_en_flows_and_prices[key]['decay']) ** (year - 1)
                r2_i += self.annual_en_flows_and_prices[key]['self_cons'] * self.annual_en_flows_and_prices[key]['price_buy'] * (
                        1 - self.annual_en_flows_and_prices[key]['decay']) ** (year - 1)
                c1_i+= self.annual_en_flows_and_prices[key]['purchased'] * self.annual_en_flows_and_prices[key]['price_buy']

            c4_i = r1_i * tax_rate

            for component in self.components:
                start,end=component.inc_start_end
                is_in_range = start <= year <= end
                if is_in_range:
                    r3_i += component.inc_year
                for key in component.other_rev:
                    start,end=component.other_rev[key]['dur']
                    is_in_range = start <= year <= end
                    if is_in_range:
                        r4_i += component.other_rev[key]['unit']*component.other_rev[key]['rev_unit']
                    start, end = component.other_cost[key]['dur']
                    is_in_range = start <= year <= end
                    if is_in_range:
                        c5_i += component.other_cost[key]['unit'] * component.other_cost[key]['cost_unit']



            r1[year] = r1_i
            r2[year] = r2_i
            r3[year] = r3_i
            r4[year] = r4_i
            c1[year] = c1_i
            c2[year] = c2_i
            c3[year] = c3_i
            c4[year] = c4_i
            c5[year] = c5_i


            outflow[year] = c1_i + c2_i + c3_i + c4_i+c5_i
            inflow[year] = r1_i + r2_i + r3_i + r4_i
            cashflow[year] = inflow[year] - outflow[year]
            cashflow_cum[year] = cashflow_cum[year - 1] + cashflow[year]

        NPV = npf.npv(int_rate, cashflow)
        pbp = outflow0 / np.mean(cashflow[1:])

        ec_perf={}
        ec_perf['NPV'] = NPV
        ec_perf['pbp'] = pbp
        ec_perf['capex']=outflow0
        ec_perf['rev_from_sale'] = r1
        ec_perf['rev_savings'] = r2
        ec_perf['rev_incentives'] = r3
        ec_perf['rev_others'] = r4
        ec_perf['cost_resources'] = c1
        ec_perf['cost_opex'] = c2
        ec_perf['cost_taxes'] = c3
        ec_perf['cost_taxes_on_sale'] = c4
        ec_perf['cost_others'] = c5

        return ec_perf


