"""
Created on June 7 08:00:00 2025

@author: isabella pizzuti
"""
from src.rec_sim.Consumer import Consumer
from src.rec_sim.Prosumer import Prosumer
from src.rec_sim.Rec import Rec
from src.rec_sim.Bess import Bess
from src.rec_sim.PvPanels import PvPanels
import yaml
import pvlib
import pandas as pd
import numpy as np
from pathlib import Path
import re

def time_step_to_hour_fraction(time_step):
    match = re.match(r'(\d+)\s*min', time_step.lower())
    if match:
        minutes = int(match.group(1))
        return minutes / 60
    match = re.match(r'(\d+)\s*h', time_step.lower())
    if match:
        hours = int(match.group(1))
        return float(hours)
    raise ValueError("Unrecognized time_step format")

def run(file_path,output_dir,base_path=None):

    #read yaml docs
    if base_path:
        file_path = Path(base_path) / file_path
    file_path = file_path.resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"YAML File not found: {file_path}")
    with open(file_path, "r") as file:
        config_data = yaml.safe_load(file)

    # read simulation parameters
    time_step=time_step_to_hour_fraction(time_step=config_data["simulation"]["time_step"])
    start_date = config_data["simulation"]["start_date"]

    #read demand curve docs
    file_path = config_data["simulation"]["demand_curve_file"]
    if base_path:
        file_path = Path(base_path) / file_path
    file_path = file_path.resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"YAML File not found: {file_path}")


    df = pd.read_csv(file_path, sep=';')

    #generate system
    systems = {}
    irradiation_data = {}
    for system in config_data["systems"]:
        for sys_id, sys_conf in system.items():
            tech = sys_conf["tech"]
            economics = sys_conf["economics"]

            lat = tech["lat"]
            lon = tech["lon"]
            tilt = tech["tilt"]
            azimuth= tech["azimuth"]
            irr = pvlib.iotools.pvgis.get_pvgis_hourly(
                lat, lon, start=2019, end=2019,
                raddatabase='PVGIS-SARAH2',
                surface_tilt=tilt,
                surface_azimuth=azimuth,
                outputformat='csv',
                pvcalculation=True,
                peakpower=0.4,
                loss=14,
                url='https://re.jrc.ec.europa.eu/api/v5_2/'
            )[0]

            I_beam, I_skydiff, I_grounddiff, t_amb = [], [], [], []
            for e, e1, e2, e3 in zip(irr['poa_direct'], irr['poa_sky_diffuse'], irr['poa_ground_diffuse'], irr['temp_air']):
                I_beam.extend([e] * int(1 / time_step))
                I_skydiff.extend([e1] * int(1 / time_step))
                I_grounddiff.extend([e2] * int(1 / time_step))
                t_amb.extend([e3] * int(1 / time_step))

            irradiation_data[sys_id] = (I_beam, I_skydiff, I_grounddiff, t_amb)

            systems[sys_id] = PvPanels(
                id=sys_id,
                cap_cost=economics["cap_cost"],
                opex_cost=economics["opex_cost"],
                inc_year=economics["inc_year"],
                inc_start_end=economics["inc_start_end"],
                tax_year=economics["tax_year"],
                other_cost=economics["other_cost"],
                other_rev=economics["other_rev"],
                n_series=tech["n_series"],
                n_parallel=tech["n_parallel"],
            )

    for sys_id, obj in systems.items():
        for system in config_data["systems"]:
            if sys_id in system:
                tilt = system[sys_id]["tech"]["tilt"]
                break

        I_beam, I_skydiff, I_grounddiff, t_amb = irradiation_data[sys_id]
        obj.compute_output(slope=tilt, theta=None, I_beam=I_beam, I_skydiff=I_skydiff,
                           I_grounddiff=I_grounddiff, t_amb=t_amb)

    # generate consumers
    consumers = {}
    for cons in config_data["users"]:
        for cons_id, cons_conf in cons.items():
            dem={}
            for carrier in cons_conf["carriers"]:
                column=cons_conf["carriers"][carrier]["column"]
                dem[carrier] =  df[column]

            consumers[cons_id] = Consumer(
                id=cons_conf["id"],dem=dem)

    #generate bess
    bess_storage = {}
    for b in config_data["bess"]:
        for bess_id, bess_conf in b.items():
            tech = bess_conf["tech"]
            econ = bess_conf["economics"]

            bess_storage[bess_id] = Bess(
                id=tech["id"],
                carriers=["electricity"],
                cap_module=tech["cap_module"],
                v=tech["v"],
                i_max=tech["i_max"],
                i_min=tech["i_min"],
                soc_in=tech["soc_in"],
                soc_max=tech["soc_max"],
                soc_min=tech["soc_min"],
                n_series=tech["n_series"],
                n_parallel=tech["n_parallel"],
                cap_cost=econ["cap_cost"],
                opex_cost=econ["opex_cost"],
                inc_year=econ["inc_year"],
                inc_start_end=econ["inc_start_end"],
                tax_year=econ["tax_year"],
                other_cost=econ["other_cost"],
                other_rev=econ["other_rev"]
            )

    #generate prosumers
    prosumers = {}
    for pros in config_data["prosumers"]:
        for pros_id, pros_conf in pros.items():
            tech = pros_conf["tech"]
            econ = pros_conf["economics"]

            consumer_ids = tech["users"]
            system_ids = tech["systems"]
            bess_ids = tech["bess"]

            prosumer_systems = [systems[sid] for sid in system_ids]
            prosumer_bess = [bess_storage[bid] for bid in bess_ids]
            prosumer_consumers = [consumers[cid] for cid in consumer_ids]
            prosumers[pros_id] = Prosumer(
                id=tech["id"],
                users=prosumer_consumers,
                systems=prosumer_systems ,
                bess= prosumer_bess,
                carriers=tech["carriers"]
            )

            pros_obj=prosumers[pros_id]
            pros_obj.energy_performance(time=time_step)


            flows_and_prices={}
            for carrier in tech['carriers']:

                flows_and_prices[carrier] =  {
                        "sold": sum(pros_obj.en_perf_evolution[carrier]["surplus"]) / 1000 * time_step,
                        "self_cons": sum(pros_obj.en_perf_evolution[carrier]["self_cons"]) / 1000 * time_step,
                        "purchased": 0,
                        "price_sold":  econ['carriers_and_costs'][carrier]['price_sold'],
                        "price_buy":   econ['carriers_and_costs'][carrier]['price_buy'],
                        "decay": econ['carriers_and_costs'][carrier]['decay']
                    }

            pros_obj.economic_performance(
                time_horizon=config_data['simulation']["time_horizon"],
                tax_rate=econ["tax_rate"],
                int_rate=econ["int_rate"],
                other_capex_perc=econ["other_capex_perc"],
                annual_en_flows_and_price=flows_and_prices
            )

    #generate recs
    recs = {}
    for rec in config_data["rec"]:
        for rec_id, rec_conf in rec.items():
            tech = rec_conf["tech"]
            econ = rec_conf["economics"]
            rec_prosumers = [prosumers[pid] for pid in tech["prosumers"]]
            rec_consumers = [consumers[cid] for cid in tech["consumers"]]
            rec_systems = [systems[sid] for sid in tech["rec_systems"]]
            rec_bess = [bess_storage[bid] for bid in tech["bess"]]

            recs[rec_id] = Rec(
                id=tech["id"],
                prosumers=rec_prosumers,
                consumers=rec_consumers,
                rec_systems=rec_systems,
                rec_bess=rec_bess,
                carriers=tech["carriers"]
            )

            rec_obj=recs[rec_id]

            out_en_rec = rec_obj.energy_performance(time=time_step)

            flows_and_prices = {}
            for carrier in tech['carriers']:
                flows_and_prices[carrier] = {
                    "sold": sum(rec_obj.en_perf_evolution[carrier]["prod_rec"]) / 1000 * time_step,
                    "self_cons": sum(rec_obj.en_perf_evolution[carrier]["shared"]) / 1000 * time_step,
                    "purchased": 0,
                    "price_sold": econ['carriers_and_costs'][carrier]['price_sold'],
                    "price_buy": econ['carriers_and_costs'][carrier]['price_buy'],
                    "decay": econ['carriers_and_costs'][carrier]['decay']
                }


            out_ec_rec = rec_obj.economic_performance(
                time_horizon=config_data['simulation']["time_horizon"],
                tax_rate=econ["tax_rate"],
                int_rate=econ["int_rate"],
                other_capex_perc=econ["other_capex_perc"],
                annual_en_flows_and_price=flows_and_prices
            )

    #generate csv files
    rec_dfs = []
    for rec_id, rec_obj in recs.items():
        for carrier, perf_dict in rec_obj.en_perf_evolution.items():
            temp_df = pd.DataFrame(perf_dict)
            temp_df.columns = [f"{rec_id}_{carrier}_{col}" for col in temp_df.columns]
            rec_dfs.append(temp_df)
    rec_result = pd.concat(rec_dfs, axis=1)

    n_rows = rec_result.shape[0]
    timeline = pd.date_range(
        start=start_date,
        periods=n_rows,
        freq=config_data["simulation"]["time_step"]
    )
    rec_result.insert(0, 'date', timeline)
    rec_result.to_csv(f'{output_dir}/recs_en_perf_evolution_kW.csv', index=False)
    pros_dfs = []
    for pros_id, pros_obj in prosumers.items():
        for carrier, perf_dict in pros_obj.en_perf_evolution.items():
            temp_df = pd.DataFrame(perf_dict)
            temp_df.columns = [f"{pros_id}_{carrier}_{col}" for col in temp_df.columns]
            pros_dfs.append(temp_df)
    pros_result = pd.concat(pros_dfs, axis=1)
    pros_result.insert(0, 'date', timeline)
    pros_result.to_csv(f'{output_dir}/prosumers_en_perf_evolution_kW.csv', index=False)


    all_data = {}
    max_len = 0
    for rec_id, rec_obj in recs.items():
        ec_perf = rec_obj.ec_perf
        for k, v in ec_perf.items():
            col_name = f"{rec_id}_{k}"
            if isinstance(v, (list, np.ndarray)):
                v = list(v)
            else:
                v = [v]
            all_data[col_name] = v
            if len(v) > max_len:
                max_len = len(v)

    for col in all_data:
        if len(all_data[col]) < max_len:
            all_data[col] += [None] * (max_len - len(all_data[col]))

    rename_map = {}
    for rec_id in recs.keys():
        old_name = f"{rec_id}_rev_savings"
        new_name = f"{rec_id}_rev_inc_on_shared"
        if old_name in all_data:
            rename_map[old_name] = new_name

    df = pd.DataFrame(all_data)
    df = df.rename(columns=rename_map)
    df.to_excel(f'{output_dir}/recs_ec_perf_€.xlsx', index=False)
    rec_result_ec=df

    all_data_pros = {}
    max_len = 0
    for pros_id, pros_obj in prosumers.items():
        ec_perf = pros_obj.ec_perf
        for k, v in ec_perf.items():
            col_name = f"{pros_id}_{k}"
            if isinstance(v, (list, np.ndarray)):
                v = list(v)
            else:
                v = [v]
            all_data_pros[col_name] = v
            if len(v) > max_len:
                max_len = len(v)

    for col in all_data_pros:
        if len(all_data_pros[col]) < max_len:
            all_data_pros[col] += [None] * (max_len - len(all_data_pros[col]))

    df = pd.DataFrame(all_data_pros)
    df.to_excel(f'{output_dir}/prosumers_ec_perf_€.xlsx', index=False)
    pros_result_ec = df
    simulation = {'time_step': time_step, 'timeline': timeline,'start_date':start_date}
    all_components = {'recs':recs,'prosumers':prosumers,'consumers':consumers,'systems':systems,'bess':bess_storage}

    return simulation,all_components,rec_result, pros_result, rec_result_ec,pros_result_ec