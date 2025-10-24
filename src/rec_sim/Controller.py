"""
Created on June 5 08:00:00 2025

@author: isabella pizzuti
"""

import numpy as np

class Controller:
    def __init__(self, bess):
        """
        :param bess--> list of obj by Bess
        """
        self.bess = bess

    def energy_performance(self, production, demand, time):
        """
             :param production: DataSeries or Array-->  (kW)
             :param demand: DataSeries or Array--> (kW)
             :param time: float--> 1 if hourly analysis, 0.25 if quarterly analysis
             :return:
                    stored_tot_ev (kW)
                    supply_tot_ev (kW)
                    surplus_tot_ev (kW)
                    deficit_tot_ev (kW)
                    power_tot_ev (kW)
                    soc_tot_ev

             """


        stored_tot_ev = np.zeros(len(production))
        supply_tot_ev = np.zeros(len(production))
        power_tot_ev = np.zeros(len(production))
        surplus_tot_ev = np.zeros(len(production))
        deficit_tot_ev = np.zeros(len(production))
        soc_tot_ev = np.zeros(len(production))
        bess = self.bess
        len_ref = len(production)
        for battery in bess:
            battery.en_perf_evolution['power_in'] = np.zeros(len_ref)
            battery.en_perf_evolution['soc'] = np.zeros(len_ref)
            battery.en_perf_evolution['stored'] = np.zeros(len_ref)
            battery.en_perf_evolution['supply'] = np.zeros(len_ref)
            battery.en_perf_evolution['power'] = np.zeros(len_ref)
            battery.en_perf_evolution['surplus'] = np.zeros(len_ref)
            battery.en_perf_evolution['deficit'] = np.zeros(len_ref)
            battery.en_perf_evolution['current'] = np.zeros(len_ref)
            battery.en_perf_evolution['case'] = np.zeros(len_ref)

        for i in range(len(production)):
            power_in = production[i] - demand[i]
            stored_tot = 0
            supply_tot = 0
            power_tot = 0
            if power_in > 0:
                bess_sorted = sorted(bess, key=lambda bess: bess.soc_in)
                for battery in bess_sorted:
                    power_in_battery=power_in
                    power_in_battery1, soc, stored, supply, power, surplus, deficit, current, case = battery.energy_performance(
                        power_in_battery, time)
                    battery.soc_in = soc
                    power_in -= stored
                    stored_tot += stored
                    supply_tot += supply
                    power_tot = stored_tot
                    memories = {'power_in': power_in, 'soc': soc, 'stored': stored, 'supply': supply,
                                'power': power,
                                'surplus': surplus,
                                'deficit': deficit, 'current': current, 'case': case}
                    for key, value in memories.items():
                        battery.en_perf_evolution[key][i] = value
            else:
                bess_sorted = sorted(bess, key=lambda bess: bess.soc_in, reverse=True)
                for battery in bess_sorted:
                    power_in_battery1, soc, stored, supply, power, surplus, deficit, current, case = battery.energy_performance(
                        power_in, time)
                    battery.soc_in = soc

                    power_in += supply
                    stored_tot += stored
                    supply_tot += supply
                    power_tot = -supply_tot
                    memories = {'power_in': power_in, 'soc': soc, 'stored': stored, 'supply': supply,
                                'power': power,
                                'surplus': surplus,
                                'deficit': deficit, 'current': current, 'case': case}
                    for key, value in memories.items():
                        battery.en_perf_evolution[key][i] = value


            num=0
            den=0
            for battery in bess:
                num+=battery.en_perf_evolution['soc'][i]*battery.cap
                den+=battery.cap

            soc_tot=num/den

            stored_tot_ev[i] = stored_tot
            supply_tot_ev[i] = supply_tot
            surplus_tot_ev[i] = production[i] - min(production[i], demand[i]) - stored_tot
            deficit_tot_ev[i] = demand[i] - min(production[i], demand[i]) - supply_tot
            power_tot_ev[i] = power_tot
            soc_tot_ev[i]=soc_tot



        return stored_tot_ev, supply_tot_ev, power_tot_ev, surplus_tot_ev, deficit_tot_ev,soc_tot_ev


