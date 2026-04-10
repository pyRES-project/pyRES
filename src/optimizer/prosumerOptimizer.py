"""
Created on June 7 08:00:00 2023

@author: isabella pizzuti
"""
import copy
import math
# Relative path
import sys
import os

path_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(path_dir)

# import PyREC module
from Library.Utility.Integrator import integrator
from Library.RecMembers.Consumer import Consumer
from Library.RecMembers.Prosumer import Prosumer
from Library.ProductionSystem.Electricity.WeatherDependent.PvPanels import PvPanels
from Library.AuxiliaryComponent.AuxiliaryComponent import AuxiliaryComponent
from Library.Storage.Battery.Bess import Bess

# import other modules
import pandas as pd
import numpy as np
import pvlib
import matplotlib.pyplot as plt
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.termination import get_termination
from pymoo.optimize import minimize

# generate graphs and summary in file.xlsx
graph = True
excel_file = True

# set simulation time step
time = 0.25  # 0.25 for quarter-hour analysis- 1 for hourly analysis

# import weather data for weather-dependent technologies
irradiation = pvlib.iotools.pvgis.get_pvgis_hourly(42.416, 13.079, start=2019, end=2019,
                                                   raddatabase='PVGIS-SARAH2', surface_tilt=30, surface_azimuth=0,
                                                   outputformat='csv', pvcalculation=True, peakpower=0.4, loss=14,
                                                   url='https://re.jrc.ec.europa.eu/api/v5_2/')
I_beam0 = irradiation[0]['poa_direct']  #
I_skydiff0 = irradiation[0]['poa_sky_diffuse']
I_grounddiff0 = irradiation[0]['poa_ground_diffuse']
t_amb0 = irradiation[0]['temp_air']

## create the quarter-hour radiation curves
I_beam = []
I_skydiff = []
I_grounddiff = []
t_amb = []
for e, e1, e2, e3 in zip(I_beam0, I_skydiff0, I_grounddiff0, t_amb0):
    I_beam.extend([e] * int(1 / time))
    I_skydiff.extend([e1] * int(1 / time))
    I_grounddiff.extend([e2] * int(1 / time))
    t_amb.extend([e3] * int(1 / time))


# input and independent variables of optimization
demand_curve = pd.read_excel('DemandCurve/demand_qh_kw_el.xlsx')
df_price_ev_qh=pd.read_excel('PUNCurve/price_ev_qh_gme.xlsx')
price_ev_qh=df_price_ev_qh['€/MWh']
prosumer_list=['Supermarket']#,'Residential2','Supermarket','Restaurant','School','Office']
space_max_list=[720]#,100,720,800,300,150]
utilization_factor = 0.6  # portion of space where solar panels can be installed
dim_panel = 1.65  # [m2]
cap_panel = 0.4  # [kW]
cap_battery_nom=2.560
n_series_max=10
n_parallel_max=10

mode_controller=1
price_threshold=130
price_threshold1=60
min_self_cons_ratio=0.3  # minimum self-consumption ratio (self_cons/dem) constraint
boundaries_collection={}

#start optimization
for i in range(len(prosumer_list)):
    # dependent variables of optimization
    space_max = space_max_list[i]  # [m2] The maximum available space for the installation of photovoltaic panels.
    cap_max = int(space_max * utilization_factor / dim_panel * cap_panel)
    cap_max_battery = cap_max * 3
    n_panels_max = int(cap_max / cap_panel)
    power_peak = demand_curve[prosumer_list[i]].max()

    while n_series_max * n_parallel_max * cap_battery_nom > cap_max_battery:
        n_series_max -= 1
        n_parallel_max -= 1

    n_modules_max = round(cap_max_battery / (n_series_max * n_parallel_max * cap_battery_nom))

    # define consumers
    consumer = Consumer(id=prosumer_list[i], dem=demand_curve[prosumer_list[i]],
                         carrier=['electricity'])

    boundaries_collection[prosumer_list[i]]=[cap_max,n_series_max,n_parallel_max,n_modules_max]

    class MyProblem(ElementwiseProblem):

        def __init__(self):
            super().__init__(n_var=4,
                             n_obj=2,
                             n_constr=3,
                             xl=np.array([1,0,1,1], dtype=int), xu=np.array([n_panels_max,n_series_max,n_parallel_max,n_modules_max], dtype=int), type_var=int)

        def _evaluate(self, X, out, *args, **kwargs):
            pv_power = X[0] * cap_panel
            print(X)
            # cost base on scale economy
            if pv_power <= 20:
                cost_kW_pv = 1400
                cost_oem_pv=40
                cost_inverter=185
            elif pv_power <= 200:
                cost_kW_pv = 1100
                cost_oem_pv=10
                cost_inverter = 90
            else:
                cost_kW_pv = 950
                cost_oem_pv = 5
                cost_inverter = 45

            cap_battery=X[1]*X[2]*X[3]*cap_battery_nom

            if cap_battery <= 20:
                cost_kWh_battery = 720
            elif cap_battery <= 200:
                cost_kWh_battery = 520
            else:
                cost_kWh_battery = 320



            list_bess = []
            for i in range(int(X[3])):
                list_bess.append(Bess(id='bess{0}'.format(i), tech='bess', n_series=X[1], n_parallel=X[2],
                                      capacity=cap_battery_nom, cost_kWh=cost_kWh_battery, oem_cost_kWh=0, v=25.6,
                                      replacement_year=8, i_min=5, i_max=100, soc_max=1, soc_min=0.1, soc_in=1,
                                      bonus50per=False))

            # define auxiliary components
            inverter1 = AuxiliaryComponent(id='inverter1', tech='inverter', replacement_year=10,
                                           replacement_cost=cost_inverter*pv_power)

            # define production system
            pv1 = PvPanels(id='pv1', tech='pv', power=pv_power, life_time=20, p_con=1, cost_kW=cost_kW_pv, oem_cost_kW=cost_oem_pv,
                           n_series=X[0], n_parallel=1, aux_components=[inverter1],BonusREC=0.4)
            # cost_kW must include the increase due to the battery cost


            # calculate production from systems
            pv1.compute_output(slope=30, I_beam=I_beam, I_skydiff=I_skydiff,
                               I_grounddiff=I_grounddiff, t_amb=t_amb)

            # define prosumer
            if X[1]*X[2]*X[3]==0:
                prosumer1 = Prosumer(id='supermarket', plant=[pv1], consumer=[consumer], carrier=['electricity'], bess=None)
                out_en_prosumer1 = prosumer1.energy_perfomance(time=time)
            else:
                prosumer1 = Prosumer(id='supermarket', plant=[pv1], consumer=[consumer], carrier=['electricity'],
                                     bess=list_bess,mode_controller=mode_controller)


                out_en_prosumer1 = prosumer1.energy_perfomance(time=time, price_threshold=price_threshold,
                                                               price_threshold1=price_threshold1, price_ev=price_ev_qh,
                                                               power_peak=power_peak)



            # calculate prosumer economical perfomance
            out_ec_prosumer1 = prosumer1.economic_perfomance(time=time, t_inv=20, down_payment_percentual=100, t_res=0,
                                                             int_rate=0.03,
                                                             pr_import={'electricity': price_ev_qh},
                                                             pr_export={'electricity': price_ev_qh * 0.8}, tax=20)



            carrier = ['electricity']
            list_prosumer = [prosumer1]

            # Calculate global energy perfomance (annual,month,day)
            summary = True
            if summary == True:
                annual = {}
                month = {}
                day = {}
                for prosumer in list_prosumer:
                    annual[prosumer.id] = {}
                    month[prosumer.id] = {}
                    for car in carrier:
                        annual[prosumer.id][car] = {}
                        month[prosumer.id][car] = {}
                        for variable in prosumer.en_perf_evolution[car].keys():
                            annual[prosumer.id][car][variable] = integrator(
                                dataseries=prosumer.en_perf_evolution[car][variable], unit='power',
                                period='year') / 1000
                            month[prosumer.id][car][variable] = integrator(
                                dataseries=prosumer.en_perf_evolution[car][variable], unit='power',
                                period='month') / 1000

            f1 = -annual[prosumer1.id]['electricity']['self_cons']
            f2 = -out_ec_prosumer1[0]


            g1 = annual[prosumer1.id]['electricity']['prod'] / annual[prosumer1.id]['electricity']['dem'] - 2
            g2 = -out_ec_prosumer1[0]
            g3 = min_self_cons_ratio - annual[prosumer1.id]['electricity']['self_cons'] / annual[prosumer1.id]['electricity']['dem']

            out["F"] = [f1, f2]
            out["G"] = [g1, g2, g3]

    vectorized_problem = MyProblem()
    problem = MyProblem()
    algorithm = NSGA2(
        pop_size=100,
        n_offsprings=50,
        sampling=IntegerRandomSampling(),  # int_random
        crossover=SBX(prob=0.9, eta=15, repair=None),  # SBX supporta int arrays se type_var=int
        mutation=PM(prob=0.9, eta=20),  # PM supporta int arrays se type_var=int
        eliminate_duplicates=True
    )
    termination = get_termination("n_gen", 10)

    res = minimize(problem,
                   algorithm,
                   termination,
                   seed=1,
                   save_history=True,
                   verbose=True)

    X = res.X
    F = -res.F


    dfoutput1 = pd.DataFrame(data=X)
    dfoutput2 = pd.DataFrame(data=F)
    dfoutput = pd.concat([dfoutput1, dfoutput2], axis=1)

    dfoutput.to_excel('OutputJournal/Sizing_Nsga_{0}'.format(prosumer_list[i]) + '.xlsx')

    plt.figure()
    plt.scatter(F[:, 0], F[:, 1] / 1000, s=60, c=X[:,0] * cap_panel, cmap='rainbow', edgecolors='k')
    plt.xlabel('Self-consumption [MWh]')
    plt.ylabel('NPV [k€] ')
    plt.yticks( )
    plt.xticks()
    cb = plt.colorbar()
    cb.set_label('PV [kWp] ')
    plt.grid()
    plt.savefig('OutputJournal/ParetoFront_{0}.png'.format(prosumer_list[i]))
    plt.show()

    plt.figure()
    plt.scatter(F[:, 0], F[:, 1] / 1000, s=60, c=X[:,1] * X[:,2]*X[:,3]*cap_battery_nom, cmap='rainbow', edgecolors='k')
    plt.xlabel('Self-consumption [MWh]')
    plt.ylabel('NPV [k€] ')
    plt.yticks( )
    plt.xticks()
    cb = plt.colorbar()
    cb.set_label('Capacity [kWh] ')
    plt.grid()
    plt.savefig('OutputJournal/ParetoFront1_{0}.png'.format(prosumer_list[i]))
    plt.show()

df=pd.DataFrame(data=boundaries_collection)
df.to_excel('OutputJournal/Boundaries' + '.xlsx')

# Plot convergence1

from pymoo.indicators.hv import Hypervolume
import numpy as np
import matplotlib.pyplot as plt

F_all = np.vstack([gen.pop.get("F") for gen in res.history])
ref_point = np.max(F_all, axis=0) + 1
hv = Hypervolume(ref_point=ref_point)
hv_values = []
n_evals = []
eval_count = 0
for gen in res.history:
    F = gen.pop.get("F")
    hv_values.append(hv.do(F))

    eval_count += len(gen.pop)
    n_evals.append(eval_count)
plt.figure()
plt.plot(n_evals, np.array(hv_values)/np.array(hv_values).max(), marker='o')
plt.xlabel("Function Evaluations")
plt.ylabel("Hypervolume")
plt.grid(True)
plt.savefig('OutputJournal/Hypervolume_Convergence_{0}.png'.format(prosumer_list[i]))
plt.show()




