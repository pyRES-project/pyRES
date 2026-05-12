"""
Created on June 5 08:00:00 2025

@author: isabella pizzuti
"""
from src.rec_sim.System import System

class Bess(System):
    def __init__(self, id, cap, cap_cost, opex_cost, inc_year, inc_start_end, tax_year,
                 c_rate=1.0,
                 soc_in=0.5, soc_max=1.0, soc_min=0.0,
                 eta_charge=0.95,
                 eta_discharge=0.95,
                 self_discharge_rate_per_hour=0.00004,
                 lifetime_years=None,
                 carriers=['electricity'],
                 other_cost={'item1': {'unit': 0, 'cost_unit': 0, 'dur': [0, 0]}},
                 other_rev={'item1': {'unit': 0, 'rev_unit': 0, 'dur': [0, 0]}}):

        """
        :param id:  str--> id code
        :param cap: float --> total battery capacity (kWh)
        :param cap_cost: float --> initial cost (euro/kWh)
        :param opex_cost: float --> operating cost (euro/kWh)
        :param inc_year: float --> incentives (euro/year)
        :param inc_start_end:  list --> start and end date in year e.g. [1,6]
        :param tax_year: float --> taxes (euro/year)
        :param other_cost: dict--> e.g. {'item1': {'unit': 0, 'cost_unit': 0, 'dur': [0, 0]}}
        :param other_rev:  dict--> e.g. {'item1': {'unit': 0, 'cost_unit': 0, 'dur': [0, 0]}}
        :param c_rate: float --> maximum C-rate (1.0 = full charge/discharge in 1h, 0.5 = in 2h)
        :param soc_in: float --> initial state of charge
        :param soc_max: float --> max state of charge
        :param soc_min: float --> min state of charge
        :param eta_charge: float --> charging efficiency (0-1, typical 0.92-0.97)
        :param eta_discharge: float --> discharging efficiency (0-1, typical 0.92-0.97)
        :param self_discharge_rate_per_hour: float --> self-discharge rate per hour (typical 0.00004 for Li-ion)
        :param lifetime_years: int --> component lifetime in years for replacement cost (None = no replacement)
        """

        if cap <= 0:
            raise ValueError(f"BESS '{id}': cap must be > 0, got {cap}")
        if c_rate <= 0:
            raise ValueError(f"BESS '{id}': c_rate must be > 0, got {c_rate}")
        if soc_min < 0 or soc_max > 1:
            raise ValueError(f"BESS '{id}': soc_min must be >= 0 and soc_max <= 1, got [{soc_min}, {soc_max}]")
        if soc_min >= soc_max:
            raise ValueError(f"BESS '{id}': soc_min ({soc_min}) must be < soc_max ({soc_max})")
        if not soc_min <= soc_in <= soc_max:
            raise ValueError(f"BESS '{id}': soc_in ({soc_in}) must be in [soc_min={soc_min}, soc_max={soc_max}]")
        if not 0 < eta_charge <= 1:
            raise ValueError(f"BESS '{id}': eta_charge must be in (0, 1], got {eta_charge}")
        if not 0 < eta_discharge <= 1:
            raise ValueError(f"BESS '{id}': eta_discharge must be in (0, 1], got {eta_discharge}")

        self.id = id
        self.cap = cap
        self.opex = cap
        self.c_rate = c_rate
        self.soc_in = soc_in
        self.soc_max = soc_max
        self.soc_min = soc_min
        self.eta_charge = eta_charge
        self.eta_discharge = eta_discharge
        self.self_discharge_rate_per_hour = self_discharge_rate_per_hour
        self.en_perf_evolution = {}
        self.cumulative_discharge_energy = 0.0

        super().__init__(id=id, cap=self.cap, carriers=carriers, cap_cost=cap_cost,
                         opex=self.opex, opex_cost=opex_cost,
                         inc_year=inc_year, inc_start_end=inc_start_end, tax_year=tax_year,
                         other_cost=other_cost, other_rev=other_rev,
                         lifetime_years=lifetime_years)

    def energy_performance(self, power_in, time):
        """
        :param power_in: float --> input power to battery (kW), (+) charge (-) discharge
        :param time: float --> 1 if hourly analysis, 0.25 if quarterly analysis
        :return:
                power_in --> power sent to the bess (+) charge (-) discharge (kW)
                soc: float --> new state of charge
                power_from_source: float --> power power_from_source into the battery (kW)
                supply: float --> power supplied by the battery (kW)
                power: float --> power exchange with the battery (+) charge (-) discharge (kW)
                surplus: float --> surplus production (kW)
                deficit: float --> deficit (kW)
        """

        soc = self.soc_in

        # Self-discharge
        soc = max(soc * (1 - self.self_discharge_rate_per_hour * time), self.soc_min)

        power_max = self.c_rate * self.cap  # kW

        if power_in > 0:  # CHARGE
            available = self.cap * (self.soc_max - soc) / time
            desired = power_in * self.eta_charge
            charge_power = min(desired, available, power_max)
            source_power = charge_power / self.eta_charge

            power_from_source = source_power
            supply = 0
            surplus = power_in - source_power
            deficit = 0
            battery = charge_power * time

        elif power_in < 0:  # DISCHARGE
            demand_power = -power_in
            available = self.cap * (soc - self.soc_min) / time
            desired = demand_power / self.eta_discharge
            discharge_power = min(desired, available, power_max)
            delivered = discharge_power * self.eta_discharge

            power_from_source = 0
            supply = delivered
            surplus = 0
            deficit = demand_power - delivered
            battery = -discharge_power * time

        else:  # IDLE
            power_from_source = supply = surplus = deficit = 0
            battery = 0

        if battery < 0:
            self.cumulative_discharge_energy += abs(battery)

        soc = soc + battery / self.cap
        power = battery / time if time > 0 else 0

        return power_in, soc, power_from_source, supply, power, surplus, deficit

    @property
    def full_cycle_equivalents(self):
        """
        Full Cycle Equivalents (FCE) = total discharged energy / nominal capacity.

        :return: float --> number of full cycle equivalents
        """
        return self.cumulative_discharge_energy / self.cap if self.cap > 0 else 0.0
