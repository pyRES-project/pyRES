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
        :param annual_en_flows_and_prices: dict : e.g. annual_en_flows_and_prices={
            'electricity': {
                'sold': 100,        # MWh/year sold to the grid (year 1 baseline)
                'self_cons': 200,   # MWh/year self-consumed (year 1 baseline)
                'purchased': 50,    # MWh/year purchased from the grid (year 1 baseline)
                'price_sold': 104,  # €/MWh selling price
                'price_buy': 130,   # €/MWh buying price (also used for self-cons savings)
                'decay': 0.02,      # annual price decay rate
                'prod_degradation': 0.005,  # annual production degradation (PV aging)
            }
        }
        """

        self.components = components
        self.annual_en_flows_and_prices = annual_en_flows_and_prices

    def compute_cashflow(self,time_horizon,tax_rate ,int_rate, other_capex_perc=[0]):
        """

        :param time_horizon: int : investment time horizon (year)
        :param tax_rate: float: tax on revenues from sale e.g 0.2
        :param int_rate: float: interest rate for calculating NPV e.g 0.03
        :param other_capex_perc: list of other capex as percentage of total capex e.g [0.2,0.5]
        :return: ec_perf: dict : e.g. ec_perf={'NPV':value,'pbp':value,'capex':value,...}
        """

        outflow = np.zeros(time_horizon + 1)
        inflow = np.zeros(time_horizon + 1)
        cashflow = np.zeros(time_horizon + 1)
        cashflow_cum = np.zeros(time_horizon + 1)
        r1 = np.zeros(time_horizon + 1)
        r2 = np.zeros(time_horizon + 1)
        r3 = np.zeros(time_horizon + 1)
        r4 = np.zeros(time_horizon + 1)
        c1 = np.zeros(time_horizon + 1)
        c2 = np.zeros(time_horizon + 1)
        c3 = np.zeros(time_horizon + 1)
        c4 = np.zeros(time_horizon + 1)
        c5 = np.zeros(time_horizon + 1)
        c6_replacement = np.zeros(time_horizon + 1)

        # Calcolo CAPEX, OPEX annuale e tasse fisse dai componenti
        outflow0 = 0
        opex_cost = 0
        tax = 0
        for component in self.components:
            outflow0 += component.cap_cost_unit * component.cap
            opex_cost += component.opex_cost * component.opex
            tax += component.tax_year

        total_percentage = sum(other_capex_perc)
        outflow0 = outflow0 / (1 - total_percentage)

        investment_cost = outflow0

        # [FIX #3] OPEX e tax NON vengono più azzerati.
        # Nel codice originale, le righe "opex_cost = 0" e "tax = 0" dopo il calcolo
        # dell'investment_cost sovrascrivevano i valori appena calcolati nel loop sui
        # componenti, facendo risultare OPEX e tasse fisse sempre zero nel cashflow.
        # Ora i valori calcolati alle righe 55-57 vengono preservati e usati nel loop annuale.

        outflow[0] = investment_cost
        inflow[0] = 0
        cashflow[0] = inflow[0] - outflow[0]

        r1[0] = 0
        r2[0] = 0
        r3[0] = 0
        c1[0] = 0
        c2[0] = 0
        c3[0] = 0
        c4[0] = 0
        c5[0] = 0

        # Pre-calcolo schedule sostituzione componenti
        battery_replacement_schedule = {}
        for component in self.components:
            if component.lifetime_years is not None:
                lifetime = component.lifetime_years
                replacement_years = []
                year = lifetime
                while year <= time_horizon:
                    replacement_years.append(year)
                    year += lifetime
                if replacement_years:
                    replacement_cost = component.cap_cost_unit * component.cap
                    battery_replacement_schedule[component.id] = {
                        'years': replacement_years,
                        'cost': replacement_cost
                    }

        for year in range(1, time_horizon + 1):
            r1_i = 0
            r2_i = 0
            r3_i = 0
            r4_i = 0
            c1_i = 0
            # [FIX #3] OPEX e tasse fisse ora correttamente assegnati dai valori
            # calcolati nel loop sui componenti, non più zero.
            c2_i = opex_cost
            c3_i = tax
            c5_i = 0

            for key in self.annual_en_flows_and_prices:
                flow = self.annual_en_flows_and_prices[key]
                price_decay = (1 - flow['decay']) ** (year - 1)

                # [FIX #4] Degradazione della produzione PV: i flussi energetici
                # (sold, self_cons) decrescono annualmente per invecchiamento dei moduli.
                # Il parametro 'prod_degradation' rappresenta il tasso di degradazione
                # annuale della produzione (es. 0.005 = 0.5%/anno per c-Si).
                # Se non fornito, i flussi restano costanti (comportamento retrocompatibile).
                prod_degradation = flow.get('prod_degradation', 0.0)
                prod_decay = (1 - prod_degradation) ** (year - 1)

                # Flussi energetici aggiustati per degradazione PV e BESS
                sold_year = flow['sold'] * prod_decay
                self_cons_year = flow['self_cons'] * prod_decay
                purchased_year = flow['purchased']

                # [FIX #5] Se la produzione cala per degradazione, l'energia acquistata
                # dalla rete aumenta proporzionalmente (la domanda è costante).
                # L'incremento di purchased è la differenza tra il sold+self_cons del
                # primo anno e quello degradato.
                if prod_degradation > 0:
                    original_total_prod = flow['sold'] + flow['self_cons']
                    degraded_total_prod = sold_year + self_cons_year
                    additional_purchased = original_total_prod - degraded_total_prod
                    purchased_year += additional_purchased

                # Ricavi dalla vendita di energia alla rete
                # r1 = energia venduta × prezzo di vendita × decadimento prezzo
                r1_i += sold_year * flow['price_sold'] * price_decay

                # Ricavi dal risparmio per autoconsumo (energia non acquistata dalla rete)
                # r2 = energia autoconsumata × prezzo di acquisto evitato × decadimento prezzo
                r2_i += self_cons_year * flow['price_buy'] * price_decay

                # [FIX #2] Costo dell'energia acquistata dalla rete CON decadimento prezzi.
                # Nel codice originale il decay era applicato solo ai ricavi (r1, r2) ma
                # non al costo di acquisto (c1), creando un'incoerenza: se i prezzi calano,
                # calano sia per la vendita sia per l'acquisto.
                c1_i += purchased_year * flow['price_buy'] * price_decay

            c4_i = r1_i * tax_rate

            for component in self.components:
                start, end = component.inc_start_end
                if start <= year <= end:
                    r3_i += component.inc_year
                for key in component.other_rev:
                    start, end = component.other_rev[key]['dur']
                    if start <= year <= end:
                        r4_i += component.other_rev[key]['unit'] * component.other_rev[key]['rev_unit']
                    start, end = component.other_cost[key]['dur']
                    if start <= year <= end:
                        c5_i += component.other_cost[key]['unit'] * component.other_cost[key]['cost_unit']

            # Costo sostituzione batteria
            c6_i = 0
            for batt_id, schedule in battery_replacement_schedule.items():
                if year in schedule['years']:
                    c6_i += schedule['cost']
            c6_replacement[year] = c6_i

            r1[year] = r1_i
            r2[year] = r2_i
            r3[year] = r3_i
            r4[year] = r4_i
            c1[year] = c1_i
            c2[year] = c2_i
            c3[year] = c3_i
            c4[year] = c4_i
            c5[year] = c5_i

            outflow[year] = c1_i + c2_i + c3_i + c4_i + c5_i + c6_i
            inflow[year] = r1_i + r2_i + r3_i + r4_i
            cashflow[year] = inflow[year] - outflow[year]
            cashflow_cum[year] = cashflow_cum[year - 1] + cashflow[year]

        NPV = npf.npv(int_rate, cashflow)
        pbp = outflow0 / np.mean(cashflow[1:])

        ec_perf = {}
        ec_perf['NPV'] = NPV
        ec_perf['pbp'] = pbp
        ec_perf['capex'] = outflow0
        ec_perf['rev_from_sale'] = r1
        ec_perf['rev_savings'] = r2
        ec_perf['rev_incentives'] = r3
        ec_perf['rev_others'] = r4
        ec_perf['cost_resources'] = c1
        ec_perf['cost_opex'] = c2
        ec_perf['cost_taxes'] = c3
        ec_perf['cost_taxes_on_sale'] = c4
        ec_perf['cost_others'] = c5
        ec_perf['cost_bess_replacement'] = c6_replacement

        return ec_perf
