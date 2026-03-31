import pandas as pd
import matplotlib.pyplot as plt
import scipy
from scipy.interpolate import interp1d
import numpy as np

# import pyREC module
import sys
import os

path_dir=os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(path_dir)


#pyREC module
from pyRECs.Library.RecMembers.User import User
from pyRECs.Library.RecMembers.Prosumer import Prosumer
from pyRECs.Library.Utility.Integrator import integrator
from pyRECs.Library.Utility.Constants import *
from pyRECs.Library.RecMembers.Rec import Rec
from pyRECs.Library.ProductionSystem.Electricity.WeatherDependent.PvPanels import PvPanels
from pyRECs.Library.Fuel.Fuel import Fuel
from pyRECs.Library.ProductionSystem.Electricity.Flexible.InternalCombustionEngine import InternalCombustionEngine
from pyRECs.Library.AuxiliaryComponent.BiomassGasifier import BiomassGasifier
from pyRECs.Library.AuxiliaryComponent.HeatRecoverySystem import HeatRecoverySystem
from pyRECs.Library.AuxiliaryComponent.GasStorage import GasStorage
from pyRECs.Library.AuxiliaryComponent.GasCompressor import GasCompressor
from pyRECs.Library.AuxiliaryComponent.AuxiliaryComponent import AuxiliaryComponent
from pyRECs.Library.ProductionSystem.CHP.BiomassSystem import BiomassSystem
#Optimizer module
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.factory import get_termination
from pymoo.optimize import minimize
# simulation parameter
time = 0.25

graph = True


# input data
data_meteo = pd.read_csv('Input/DataMeteo/datameteo.csv', delimiter=',')
load_el = pd.read_csv('Input/user_qc_kW_el.csv', delimiter=';')
load_heat = pd.read_csv('Input/user_qc_kW_heat.csv', delimiter=';')
df_biomasssystem1=pd.read_csv('Input/TestBiomassModel/BiomassSystem_input.csv', delimiter=';')
df=pd.read_csv('Input/TestBiomassModel/Compressor_input.csv', delimiter=';')
t_input_storage=df['t_input']
ice_curve_1200=pd.read_csv('Input/Ice_curve_input_1200.csv', delimiter=';')
ice_curve_600=pd.read_csv('Input/Ice_curve_input_600.csv', delimiter=';')
ice_curve_300=pd.read_csv('Input/Ice_curve_input_300.csv', delimiter=';')
ice_curve_200=pd.read_csv('Input/Ice_curve_input_200.csv', delimiter=';')
ice_curve_100=pd.read_csv('Input/Ice_curve_input_100.csv', delimiter=';')
ice_curve_50=pd.read_csv('Input/Ice_curve_input_50.csv', delimiter=';')
ice_curve_35=pd.read_csv('Input/Ice_curve_input_35.csv', delimiter=';')
ice_curve_19=pd.read_csv('Input/Ice_curve_input_19.csv', delimiter=';')

# define user
school = User(id='school', dem=[load_el['school'], load_heat['school']], pod='None', group='pubblic institue',
              plants='pv1', carrier=['electricity', 'heat'])
# pool = User(id='pool', dem=[load_el['pool'], load_heat['pool']], pod='None', group='pubblic institue', plants='pv1',
#             carrier=['electricity', 'heat'])
pmi = User(id='pmi', dem=[load_el['pmi'], load_heat['pmi']], pod='None', group='pmi', plants='pv2',
            carrier=['electricity', 'heat'])

user_potential=[]
for i in range(0,50):
    user_potential.append(User(id='user{0}'.format(i), dem=load_el['user{0}'.format(i)], pod='None', group='residential',carrier=['electricity']))



# define plant
n_series = 150
n_parallel = 1
power_pv = n_series * n_parallel * 0.4
cost_inverter_kW = 140
inverter1 = AuxiliaryComponent(id='inverter1', tech='inverter', replacement_year=10, replacement_cost=power_pv*cost_inverter_kW)
pv1 = PvPanels(id='pv1', pod=None, ma=None, ts=None, tech='pv', user='user0', status='new', power=power_pv, gse_mode='rid',
               decay=0.006, life_time=20, p_con=1, cost_kW=1500, oem_cost_kW=40, inc=3750, dur_inc=10, mode_mppt=1,
               isc_ref=11.32,
               voc_ref=43.8, t_cell_ref_c=25, I_tot_ref=1000,
               vmppt_ref=37.2, imppt_ref=10.76, mu_isc_ref=0.04,
               mu_voc_ref=0.24, ser_cell=60, t_cell_noct_c=44, area=1.81, n_series=n_series, n_parallel=n_parallel,
               carrier='electricity', dur_inc_kWh=0, inc_kWh=0, inc_kW=0, aux_components=[inverter1],oem_cost_kWh=0)






#calculate production plant
pv1.compute_output(slope=30, theta=None, I_beam=data_meteo['I_beam [W/m2]'], I_skydiff=data_meteo['I_skydiff [W/m2]'],
                   I_grounddiff=data_meteo['I_grounddiff [W/m2]'], t_amb=data_meteo['t_amb [C]'])


prosumer1 = Prosumer(id='School complex', plant=[pv1], user=[school],list_carrier=['electricity'])
out_prosumer1 = prosumer1.energy_perfomance(time=0.25)


power_cog = 100
n_series = 200
n_parallel = 1

# dependent variables
power_pv = n_series * n_parallel * 0.4
cost_inverter_kW_2 = 140
# define user
inverter2 = AuxiliaryComponent(id='inverter1', tech='inverter', replacement_year=10,
                               replacement_cost=power_pv * cost_inverter_kW_2)

pv2 = PvPanels(id='pv1', pod=None, ma=None, ts=None, tech='pv', user='user0', status='new', power=power_pv,
               gse_mode='rid',
               decay=0.006, life_time=20, p_con=1, cost_kW=1500, oem_cost_kW=40, inc=0, dur_inc=10, mode_mppt=1,
               isc_ref=11.32,
               voc_ref=43.8, t_cell_ref_c=25, I_tot_ref=1000,
               vmppt_ref=37.2, imppt_ref=10.76, mu_isc_ref=0.04,
               mu_voc_ref=0.24, ser_cell=60, t_cell_noct_c=44, area=1.81, n_series=n_series, n_parallel=n_parallel,
               carrier='electricity', dur_inc_kWh=0, inc_kWh=0, inc_kW=0, aux_components=[inverter2], oem_cost_kWh=0,
               bonus50per=True)

syngas2 = Fuel(fuel='syngas')
biomass2 = Fuel(fuel='wood pellets')
biomassgasifier2 = BiomassGasifier(id='biomassgasifier2', biomass_flow_rate=90, en_el_abs_rate=3 / 115,
                                   heat_developed_rate=70 / 115, temperature_syngas=25, pressure_syngas=1, eff=0.93,
                                   biomass=biomass2, syngas=syngas2, replacement_year=0, replacement_cost=0)
compressor2 = GasCompressor(id='compresssor2', tech='GasCompressor', fuel=syngas2, replacement_year=0,
                            replacement_cost=0)
storage2 = GasStorage(id='gasstorage2', capacity=10, pressure_max=200, replacement_year=0, replacement_cost=0)
heat_recovery2 = HeatRecoverySystem(id='heat2', eff=1, replacement_year=0, replacement_cost=0)
heat_recovery21 = HeatRecoverySystem(id='heat21', eff=1, replacement_year=0, replacement_cost=0)
ice2 = InternalCombustionEngine(id='ICE2', carrier='electricity', power=power_cog, power_min=power_cog * 0.4,
                                power_max=power_cog * 1.2, n_min=1, n_max=1, fuel=syngas2,
                                datasheet_curve=ice_curve_100, category='flex', cogeneration=True, p_con=1,
                                cost_kW=0, inc_kWh=0, inc_kW=0, inc=0, dur_inc=0, dur_inc_kWh=0, oem_cost_kW=0,
                                oem_cost_kWh=0, aux_components=[heat_recovery2])
biomasssystem_mode1 = BiomassSystem(id='biomasssystem1', converter=biomassgasifier2, engine=ice2, mode=1,
                                    compressor=compressor2, storage=storage2, cost_kW=4100, inc_kW=0, oem_cost_kW=0,
                                    inc_kWh=0, inc=0, dur_inc_kWh=20, dur_inc=0,
                                    initial_pressure_level=0.4, gas_temperature_storage=t_input_storage,
                                    gas_pressure_compressor=df_biomasssystem1['p'], oem_cost_kWh=0.025)

# calculate production plant

pv2.compute_output(slope=30, theta=None, I_beam=data_meteo['I_beam [W/m2]'], I_skydiff=data_meteo['I_skydiff [W/m2]'],
                   I_grounddiff=data_meteo['I_grounddiff [W/m2]'], t_amb=data_meteo['t_amb [C]'])

prosumer2 = Prosumer(id='pmi', plant=[biomasssystem_mode1,pv2], user=[pmi], list_carrier=['electricity', 'heat'])


out_prosumer2 = prosumer2.energy_perfomance(time=0.25)
pr_export_el=232.5*0.9 #[€/MWh]
pr_import_el=232.5 #[€/MWh]
pr_import_methane=0.75 #[€/Sm3]
kwh_term_methane=8.19 #[kWhterm] ottenuti da 1Sm3 di metano considerando un rendimento della caldaia pari a 0.91]
pr_import_heat=pr_import_methane/kwh_term_methane*conv_kWh_MWh
out_ec_prosumer2=prosumer2.economic_perfomance(time=0.25,t_inv=20,down_payment_percentual=100,t_res=0,int_rate=0.03,pr_import={'electricity':pr_import_el,'heat':pr_import_heat},pr_export={'electricity':pr_export_el,'heat':0},tax=20,value_cb=250)

out_ec_prosumer1 = prosumer1.economic_perfomance(time=0.25, t_inv=20, down_payment_percentual=100, t_res=0,
                                                 int_rate=0.03,
                                                 pr_import={'electricity': pr_import_el},
                                                 pr_export={'electricity': pr_export_el}, tax=20)





i=0
n_independent_var=50
class MyProblem(ElementwiseProblem):

    def __init__(self):
        super().__init__(n_var=50,
                         n_obj=2,
                         n_constr=1,
                         xl=np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], dtype=int), xu=np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1], dtype=int), type_var=int)

    def _evaluate(self, X, out, *args, **kwargs):
        selection = {}
        for i in range(0, n_independent_var):
            selection[user_potential[i]] = X[i]
        user_list = []
        for user in selection:
            if selection[user] == 1:
                user_list.append(user)
        print(user_list)

        rec=Rec(id='Rec',prosumer=[prosumer1,prosumer2],consumer=user_list,rec_plant=[],list_carrier=['electricity'],mode_inc=1)

        out_en_rec = rec.energy_perfomance(time=0.25)
        rec_ec=rec.economic_perfomance(time=time, t_inv=20,
                                      down_payment_percentual=100, t_res=0,
                                      int_rate=0.03,
                                      inc_shared={'electricity': 118, 'heat': 0},
                                      pr_export={'electricity': pr_export_el,
                                                 'heat': 0}, tax=20, p1=0.8,
                                      p2=0.2)
        NPV_rec=rec_ec[0]
        if user_list:
            revenue_single_member=rec_ec[5]
        else:
            revenue_single_member=0

        residual_dem=sum(out_en_rec['electricity']['unmet'])*time/1000
        f1 = -revenue_single_member
        f2 = -NPV_rec
        print(f1,f2)

        g1 = 0
        g2 = 0

        out["F"] = [f1, f2]
        out["G"] = [g1, g2]


        i += 1
        print('iter',i)
        print(NPV_rec)
        print(revenue_single_member)


vectorized_problem = MyProblem()
problem = MyProblem()

algorithm = NSGA2(pop_size=100, n_offsprings=50, sampling=get_sampling("int_random"),
                  crossover=get_crossover("int_sbx", prob=0.9, eta=15),
                  mutation=get_mutation("int_pm", prob=0.9, eta=20),
                  eliminate_duplicates=True)

termination = get_termination("n_gen",2)

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

dfoutput.to_excel('Output/Parameter7_new/Sizing_Nsga' + '.xlsx')

plt.figure()
plt.scatter(F[:, 0], F[:, 1]/1000,s=60, c=X[:,0]+X[:,1]+X[:,2]+X[:,3]+X[:,4]+X[:,5]+X[:,6]+X[:,7]+X[:,8]+X[:,9]+X[:,10]+X[:,11]+X[:,12]+X[:,13]+X[:,14]+X[:,15]+X[:,16]+X[:,17]+X[:,18]+X[:,19]+X[:,20]+X[:,21]+X[:,22]+X[:,23]+X[:,24]+X[:,25]+X[:,26]+X[:,27]+X[:,28]+X[:,29]+X[:,30]+X[:,31]+X[:,32]+X[:,33]+X[:,34]+X[:,35]+X[:,36]+X[:,37]+X[:,38]+X[:,39]+X[:,40]+X[:,41]+X[:,42]+X[:,43]+X[:,44]+X[:,45]+X[:,46]+X[:,47]+X[:,48]+X[:,49], cmap='rainbow', edgecolors='k')
plt.xlabel('Revenue for single member [€]')
plt.ylabel('NPV-REC [k€] ')
plt.yticks( )
plt.xticks()
cb = plt.colorbar()
cb.set_label('N° consumers ')
plt.grid()
plt.savefig('Output/Parameter7_new/0_Nsga.png')
plt.show()


