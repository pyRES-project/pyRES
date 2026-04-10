"""
Created on June 7 08:00:00 2025

@author: isabella pizzuti

[FIX #8] Decomposizione: la funzione monolitica run() è stata decomposta in
funzioni separate con responsabilità chiare:
- parse_config(): legge e valida la configurazione YAML
- fetch_meteo(): scarica dati meteo da PVGIS con error handling e cache locale
- build_components(): crea tutti gli oggetti simulazione dalla config
- run_simulation(): esegue le simulazioni energetiche ed economiche
- export_results(): genera i file CSV/Excel di output
- run(): funzione principale che orchestra il flusso completo (retrocompatibile)
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
import hashlib
import logging

logger = logging.getLogger(__name__)


def time_step_to_hour_fraction(time_step):
    match = re.match(r'(\d+)\s*min', time_step.lower())
    if match:
        minutes = int(match.group(1))
        return minutes / 60
    match = re.match(r'(\d+)\s*h', time_step.lower())
    if match:
        hours = int(match.group(1))
        return float(hours)
    raise ValueError(f"Unrecognized time_step format: '{time_step}'")


# ---------------------------------------------------------------------------
# [FIX #8] Step 1: Parse configuration
# ---------------------------------------------------------------------------
def parse_config(file_path, base_path=None):
    """
    Read and validate the YAML configuration file.

    :param file_path: str or Path --> path to config.yaml
    :param base_path: str or Path --> base directory for resolving relative paths
    :return: tuple (config_data, demand_df, time_step, start_date)
    """
    if base_path:
        file_path = Path(base_path) / file_path
    file_path = Path(file_path).resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"YAML config file not found: {file_path}")

    with open(file_path, "r") as file:
        config_data = yaml.safe_load(file)

    # [FIX #16] Validazione delle chiavi obbligatorie nella config
    required_sections = ['simulation', 'users', 'systems', 'bess', 'prosumers', 'rec']
    for section in required_sections:
        if section not in config_data:
            raise ValueError(f"Missing required section '{section}' in config YAML")

    sim = config_data["simulation"]
    for key in ['time_step', 'start_date', 'time_horizon', 'demand_curve_file']:
        if key not in sim:
            raise ValueError(f"Missing required key 'simulation.{key}' in config YAML")

    time_step = time_step_to_hour_fraction(time_step=sim["time_step"])
    start_date = sim["start_date"]

    # Read demand data
    demand_path = sim["demand_curve_file"]
    if base_path:
        demand_path = Path(base_path) / demand_path
    demand_path = Path(demand_path).resolve()
    if not demand_path.exists():
        raise FileNotFoundError(f"Demand CSV file not found: {demand_path}")

    demand_df = pd.read_csv(demand_path, sep=';')

    return config_data, demand_df, time_step, start_date


# ---------------------------------------------------------------------------
# [FIX #9, #8] Step 2: Fetch meteorological data with error handling and cache
# ---------------------------------------------------------------------------
def _get_cache_path(lat, lon, tilt, azimuth, cache_dir):
    """Generate a deterministic cache file path for PVGIS data."""
    key = f"{lat:.6f}_{lon:.6f}_{tilt}_{azimuth}"
    hash_key = hashlib.md5(key.encode()).hexdigest()[:12]
    return Path(cache_dir) / f"pvgis_{hash_key}.csv"


def fetch_meteo(lat, lon, tilt, azimuth, time_step_str, time_step_hours, cache_dir=None):
    """
    Fetch meteorological data from PVGIS with error handling and optional local cache.

    [FIX #9] Wraps the PVGIS HTTP call in try/except to provide a clear error message
    instead of a cryptic network crash. If cache_dir is provided, data is cached locally
    and reused on subsequent runs with the same coordinates/tilt/azimuth.

    :param lat: float --> latitude
    :param lon: float --> longitude
    :param tilt: float --> panel tilt angle (degrees)
    :param azimuth: float --> panel azimuth (degrees, pyRES convention: 0=South)
    :param time_step_str: str --> time step string (e.g. '15min')
    :param time_step_hours: float --> time step in hours (e.g. 0.25)
    :param cache_dir: str or Path --> directory for caching PVGIS data (None = no cache)
    :return: tuple (I_beam, I_skydiff, I_grounddiff, t_amb, wind_speed, theta)
    """
    irr = None

    # [FIX #8] Try to load from local cache first
    if cache_dir is not None:
        cache_path = _get_cache_path(lat, lon, tilt, azimuth, cache_dir)
        if cache_path.exists():
            logger.info(f"Loading cached PVGIS data from {cache_path}")
            irr = pd.read_csv(cache_path, index_col=0, parse_dates=True)

    # [FIX #9] Fetch from PVGIS with error handling
    if irr is None:
        try:
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
        except Exception as e:
            raise RuntimeError(
                f"Failed to fetch PVGIS data for location ({lat}, {lon}), "
                f"tilt={tilt}, azimuth={azimuth}. "
                f"Check your internet connection or PVGIS server status. "
                f"Original error: {type(e).__name__}: {e}"
            ) from e

        # Save to cache for future runs
        if cache_dir is not None:
            cache_path = _get_cache_path(lat, lon, tilt, azimuth, cache_dir)
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            irr.to_csv(cache_path)
            logger.info(f"Cached PVGIS data to {cache_path}")

    # Interpolation and wind speed extraction
    n_substeps = int(1 / time_step_hours)

    if n_substeps > 1:
        hourly_index = irr.index
        high_res_index = pd.date_range(
            start=hourly_index[0],
            periods=len(hourly_index) * n_substeps,
            freq=time_step_str
        )
        irr_resampled = irr[['poa_direct', 'poa_sky_diffuse', 'poa_ground_diffuse', 'temp_air']].copy()
        if 'wind_speed' in irr.columns:
            irr_resampled['wind_speed'] = irr['wind_speed']
        else:
            irr_resampled['wind_speed'] = 1.0

        irr_highres = irr_resampled.reindex(
            irr_resampled.index.union(high_res_index)
        ).interpolate(method='linear').reindex(high_res_index)

        I_beam = irr_highres['poa_direct'].values
        I_skydiff = irr_highres['poa_sky_diffuse'].values
        I_grounddiff = irr_highres['poa_ground_diffuse'].values
        t_amb = irr_highres['temp_air'].values
        wind_speed = irr_highres['wind_speed'].values
        time_index = high_res_index
    else:
        I_beam = irr['poa_direct'].values
        I_skydiff = irr['poa_sky_diffuse'].values
        I_grounddiff = irr['poa_ground_diffuse'].values
        t_amb = irr['temp_air'].values
        wind_speed = irr['wind_speed'].values if 'wind_speed' in irr.columns else np.ones(len(t_amb))
        time_index = irr.index

    # Compute angle of incidence (theta) for IAM
    site_location = pvlib.location.Location(latitude=lat, longitude=lon)
    solar_position = site_location.get_solarposition(time_index)
    theta = pvlib.irradiance.aoi(
        surface_tilt=tilt,
        surface_azimuth=azimuth + 180,
        solar_zenith=solar_position['apparent_zenith'],
        solar_azimuth=solar_position['azimuth']
    ).values
    theta = np.clip(theta, 0, 90)

    return I_beam, I_skydiff, I_grounddiff, t_amb, wind_speed, theta


# ---------------------------------------------------------------------------
# [FIX #8] Step 3: Build simulation components
# ---------------------------------------------------------------------------
def build_components(config_data, demand_df, time_step_hours, cache_dir=None):
    """
    Create all simulation objects (PV, BESS, Consumers, Prosumers) from config.

    :param config_data: dict --> parsed YAML config
    :param demand_df: pd.DataFrame --> demand profiles
    :param time_step_hours: float --> time step in hours
    :param cache_dir: str or Path --> cache directory for PVGIS data
    :return: dict with 'systems', 'consumers', 'bess', 'irradiation_data'
    """
    time_step_str = config_data["simulation"]["time_step"]

    # Build PV systems
    systems = {}
    irradiation_data = {}
    for system in config_data["systems"]:
        for sys_id, sys_conf in system.items():
            tech = sys_conf["tech"]
            economics = sys_conf["economics"]

            lat, lon = tech["lat"], tech["lon"]
            tilt, azimuth = tech["tilt"], tech["azimuth"]

            meteo = fetch_meteo(lat, lon, tilt, azimuth, time_step_str, time_step_hours, cache_dir)
            irradiation_data[sys_id] = meteo

            pv_params = {
                'id': sys_id,
                'cap_cost': economics["cap_cost"],
                'opex_cost': economics["opex_cost"],
                'inc_year': economics["inc_year"],
                'inc_start_end': economics["inc_start_end"],
                'tax_year': economics["tax_year"],
                'other_cost': economics["other_cost"],
                'other_rev': economics["other_rev"],
                'n_series': tech["n_series"],
                'n_parallel': tech["n_parallel"],
            }

            optional_pv_tech_params = [
                'isc_ref', 'voc_ref', 'vmppt_ref', 'imppt_ref',
                'mu_isc_ref', 'mu_voc_ref', 'ser_cell',
                't_cell_noct_c', 't_cell_ref_c', 'I_tot_ref',
                'area', 'mode_mppt', 'eg',
                'dc_ac_efficiency', 'mismatch_loss', 'wiring_loss', 'soiling_loss',
                'annual_degradation',
            ]
            for param in optional_pv_tech_params:
                if param in tech:
                    pv_params[param] = tech[param]

            systems[sys_id] = PvPanels(**pv_params)

    # Compute PV output
    for sys_id, obj in systems.items():
        for system in config_data["systems"]:
            if sys_id in system:
                tilt = system[sys_id]["tech"]["tilt"]
                break
        I_beam, I_skydiff, I_grounddiff, t_amb, wind_speed, theta = irradiation_data[sys_id]
        obj.compute_output(slope=tilt, theta=theta, I_beam=I_beam, I_skydiff=I_skydiff,
                           I_grounddiff=I_grounddiff, t_amb=t_amb, wind_speed=wind_speed)

    # Build consumers
    consumers = {}
    for cons in config_data["users"]:
        for cons_id, cons_conf in cons.items():
            dem = {}
            for carrier in cons_conf["carriers"]:
                column = cons_conf["carriers"][carrier]["column"]
                if column not in demand_df.columns:
                    raise ValueError(f"Column '{column}' for consumer '{cons_id}' not found in demand CSV")
                dem[carrier] = demand_df[column]
            consumers[cons_id] = Consumer(id=cons_conf["id"], dem=dem)

    # Build BESS
    bess_storage = {}
    for b in config_data["bess"]:
        for bess_id, bess_conf in b.items():
            tech = bess_conf["tech"]
            econ = bess_conf["economics"]

            bess_params = {
                'id': tech["id"],
                'carriers': ["electricity"],
                'cap_module': tech["cap_module"],
                'v': tech["v"],
                'i_max': tech["i_max"],
                'i_min': tech["i_min"],
                'soc_in': tech["soc_in"],
                'soc_max': tech["soc_max"],
                'soc_min': tech["soc_min"],
                'n_series': tech["n_series"],
                'n_parallel': tech["n_parallel"],
                'cap_cost': econ["cap_cost"],
                'opex_cost': econ["opex_cost"],
                'inc_year': econ["inc_year"],
                'inc_start_end': econ["inc_start_end"],
                'tax_year': econ["tax_year"],
                'other_cost': econ["other_cost"],
                'other_rev': econ["other_rev"],
            }

            optional_bess_tech_params = [
                'eta_charge', 'eta_discharge', 'v_min',
                'self_discharge_rate_per_hour', 'c_rate_max',
                'annual_capacity_fade', 'lifetime_years', 'min_energy_threshold',
            ]
            for param in optional_bess_tech_params:
                if param in tech:
                    bess_params[param] = tech[param]

            bess_storage[bess_id] = Bess(**bess_params)

    return systems, consumers, bess_storage


# ---------------------------------------------------------------------------
# [FIX #8] Step 4: Run simulations
# ---------------------------------------------------------------------------
def run_simulation(config_data, systems, consumers, bess_storage, time_step):
    """
    Run energy and economic performance simulations for prosumers and RECs.

    :return: tuple (prosumers, recs)
    """
    # Build and simulate prosumers
    prosumers = {}
    for pros in config_data["prosumers"]:
        for pros_id, pros_conf in pros.items():
            tech = pros_conf["tech"]
            econ = pros_conf["economics"]

            prosumer_systems = [systems[sid] for sid in tech["systems"]]
            prosumer_bess = [bess_storage[bid] for bid in tech["bess"]]
            prosumer_consumers = [consumers[cid] for cid in tech["users"]]
            prosumers[pros_id] = Prosumer(
                id=tech["id"],
                users=prosumer_consumers,
                systems=prosumer_systems,
                bess=prosumer_bess,
                carriers=tech["carriers"]
            )

            pros_obj = prosumers[pros_id]
            pros_obj.energy_performance(time=time_step)

            flows_and_prices = {}
            for carrier in tech['carriers']:
                ep = pros_obj.en_perf_evolution[carrier]
                carrier_costs = econ['carriers_and_costs'][carrier]

                # [FIX #5] Calcolo dell'energia acquistata dalla rete (purchased):
                # nel codice originale era hardcoded a 0, ignorando completamente il
                # costo dell'energia non coperta dalla produzione. L'energia acquistata
                # corrisponde all'unmet demand (domanda non soddisfatta dalla produzione
                # e dal BESS), convertita da kW a MWh.
                purchased = float(np.sum(ep["unmet"])) / 1000 * time_step

                # [FIX #1] Supporto profili prezzi orari (time-of-use):
                # se nella config sono presenti 'price_sold_profile' o 'price_buy_profile'
                # come array orari/sub-orari, i ricavi/costi vengono calcolati come somma
                # pesata (energia × prezzo a ciascun timestep) anziché come prodotto
                # semplice (energia annua × prezzo medio fisso).
                # Se i profili non sono presenti, si usano i prezzi fissi (retrocompatibile).
                if 'price_sold_profile' in carrier_costs:
                    # Prezzo di vendita variabile nel tempo: ricavo = Σ(surplus_t × price_t × time_step)
                    price_sold_profile = np.array(carrier_costs['price_sold_profile'])
                    sold_revenue = float(np.sum(ep["surplus"] * price_sold_profile)) * time_step / 1000
                    # Per Economics usiamo il ricavo diretto e un prezzo medio ponderato
                    sold_mwh = float(np.sum(ep["surplus"])) / 1000 * time_step
                    avg_price_sold = sold_revenue / sold_mwh if sold_mwh > 0 else carrier_costs.get('price_sold', 0)
                else:
                    sold_mwh = float(np.sum(ep["surplus"])) / 1000 * time_step
                    avg_price_sold = carrier_costs['price_sold']

                if 'price_buy_profile' in carrier_costs:
                    # Prezzo di acquisto variabile: costo = Σ(unmet_t × price_t × time_step)
                    price_buy_profile = np.array(carrier_costs['price_buy_profile'])
                    # Calcolo del risparmio da autoconsumo pesato per prezzo orario
                    self_cons_savings = float(np.sum(ep["self_cons"] * price_buy_profile)) * time_step / 1000
                    self_cons_mwh = float(np.sum(ep["self_cons"])) / 1000 * time_step
                    avg_price_buy = self_cons_savings / self_cons_mwh if self_cons_mwh > 0 else carrier_costs.get('price_buy', 0)
                else:
                    self_cons_mwh = float(np.sum(ep["self_cons"])) / 1000 * time_step
                    avg_price_buy = carrier_costs['price_buy']

                flows_and_prices[carrier] = {
                    "sold": sold_mwh,
                    "self_cons": self_cons_mwh,
                    "purchased": purchased,
                    "price_sold": avg_price_sold,
                    "price_buy": avg_price_buy,
                    "decay": carrier_costs['decay'],
                    # [FIX #4] Tasso di degradazione della produzione PV passato a Economics
                    # per ridurre i flussi energetici anno dopo anno.
                    "prod_degradation": carrier_costs.get('prod_degradation', 0.0),
                }

            pros_obj.economic_performance(
                time_horizon=config_data['simulation']["time_horizon"],
                tax_rate=econ["tax_rate"],
                int_rate=econ["int_rate"],
                other_capex_perc=econ["other_capex_perc"],
                annual_en_flows_and_price=flows_and_prices
            )

            pros_obj.environmental_performance(
                time_horizon=config_data['simulation']["time_horizon"],
                time_step=time_step
            )

    # Build and simulate RECs
    recs = {}
    for rec in config_data["rec"]:
        for rec_id, rec_conf in rec.items():
            tech = rec_conf["tech"]
            econ = rec_conf["economics"]
            rec_prosumers = [prosumers[pid] for pid in tech["prosumers"]]
            rec_consumers = [consumers[cid] for cid in tech["consumers"]]
            rec_systems = [systems[sid] for sid in tech["rec_systems"]]
            rec_bess = [bess_storage[bid] for bid in tech["bess"]]

            rec_users = [consumers[uid] for uid in tech.get("rec_users", [])]

            recs[rec_id] = Rec(
                id=tech["id"],
                prosumers=rec_prosumers,
                consumers=rec_consumers,
                rec_systems=rec_systems,
                rec_bess=rec_bess,
                rec_users=rec_users,
                carriers=tech["carriers"]
            )

            rec_obj = recs[rec_id]
            rec_obj.energy_performance(time=time_step)

            flows_and_prices = {}
            for carrier in tech['carriers']:
                ep = rec_obj.en_perf_evolution[carrier]
                carrier_costs = econ['carriers_and_costs'][carrier]

                # Energia acquistata e autoconsumata direttamente dalla REC:
                # calcolata solo sul carico proprio della REC (rec_users).
                # Se la REC non ha carichi propri, entrambi sono 0.
                purchased = float(np.sum(ep["purchased_rec"])) / 1000 * time_step
                self_cons_mwh = float(np.sum(ep["self_cons_rec"])) / 1000 * time_step

                # [FIX #1] Supporto profili prezzi orari per REC
                if 'price_sold_profile' in carrier_costs:
                    price_sold_profile = np.array(carrier_costs['price_sold_profile'])
                    sold_mwh = float(np.sum(ep["prod_rec"])) / 1000 * time_step
                    sold_revenue = float(np.sum(ep["prod_rec"] * price_sold_profile)) * time_step / 1000
                    avg_price_sold = sold_revenue / sold_mwh if sold_mwh > 0 else carrier_costs.get('price_sold', 0)
                else:
                    sold_mwh = float(np.sum(ep["prod_rec"])) / 1000 * time_step
                    avg_price_sold = carrier_costs['price_sold']

                if 'price_buy_profile' in carrier_costs:
                    price_buy_profile = np.array(carrier_costs['price_buy_profile'])
                    if self_cons_mwh > 0:
                        buy_revenue = float(np.sum(ep["self_cons_rec"] * price_buy_profile)) * time_step / 1000
                        avg_price_buy = buy_revenue / self_cons_mwh
                    else:
                        avg_price_buy = float(np.mean(price_buy_profile))
                else:
                    avg_price_buy = carrier_costs['price_buy']

                flows_and_prices[carrier] = {
                    "sold": sold_mwh,
                    "self_cons": self_cons_mwh,
                    "purchased": purchased,
                    "price_sold": avg_price_sold,
                    "price_buy": avg_price_buy,
                    "decay": carrier_costs['decay'],
                    "prod_degradation": carrier_costs.get('prod_degradation', 0.0),
                }

            rec_obj.economic_performance(
                time_horizon=config_data['simulation']["time_horizon"],
                tax_rate=econ["tax_rate"],
                int_rate=econ["int_rate"],
                other_capex_perc=econ["other_capex_perc"],
                annual_en_flows_and_price=flows_and_prices
            )

            rec_obj.environmental_performance(
                time_horizon=config_data['simulation']["time_horizon"],
                time_step=time_step
            )

    return prosumers, recs


# ---------------------------------------------------------------------------
# [FIX #8] Step 5: Export results
# ---------------------------------------------------------------------------
def export_results(config_data, prosumers, recs, start_date, output_dir):
    """
    Generate CSV and Excel output files from simulation results.

    :return: tuple (rec_result, pros_result, rec_result_ec, pros_result_ec)
    """
    # REC energy performance CSV (exclude 'annual' summary dict)
    rec_dfs = []
    for rec_id, rec_obj in recs.items():
        for carrier, perf_dict in rec_obj.en_perf_evolution.items():
            ts_data = {k: v for k, v in perf_dict.items() if k != 'annual'}
            temp_df = pd.DataFrame(ts_data)
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

    # Prosumer energy performance CSV (exclude 'annual' summary dict)
    pros_dfs = []
    for pros_id, pros_obj in prosumers.items():
        for carrier, perf_dict in pros_obj.en_perf_evolution.items():
            ts_data = {k: v for k, v in perf_dict.items() if k != 'annual'}
            temp_df = pd.DataFrame(ts_data)
            temp_df.columns = [f"{pros_id}_{carrier}_{col}" for col in temp_df.columns]
            pros_dfs.append(temp_df)
    pros_result = pd.concat(pros_dfs, axis=1)
    pros_result.insert(0, 'date', timeline)
    pros_result.to_csv(f'{output_dir}/prosumers_en_perf_evolution_kW.csv', index=False)

    # REC economic performance Excel
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
    rec_result_ec = df

    # Prosumer economic performance Excel
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

    # Summary tables (energy + economic + environmental aggregated KPIs)
    time_step_h = time_step_to_hour_fraction(config_data['simulation']['time_step'])

    prosumer_summary = [pros_obj.summary() for pros_obj in prosumers.values()]
    rec_summary = [rec_obj.summary() for rec_obj in recs.values()]

    # Aggregated environmental (all prosumer + REC components)
    all_systems = []
    all_bess_agg = []
    total_prod_kwh = 0
    for pros_obj in prosumers.values():
        all_systems.extend(pros_obj.systems)
        all_bess_agg.extend(pros_obj.bess)
        for carrier in pros_obj.carriers:
            ep = pros_obj.en_perf_evolution.get(carrier, {})
            total_prod_kwh += float(np.sum(ep.get('prod', 0))) * time_step_h
    for rec_obj in recs.values():
        all_systems.extend(rec_obj.rec_systems)
        all_bess_agg.extend(rec_obj.rec_bess)
        for carrier in rec_obj.carriers:
            ep = rec_obj.en_perf_evolution.get(carrier, {})
            total_prod_kwh += float(np.sum(ep.get('prod_rec', 0))) * time_step_h

    from src.rec_sim.Environmentals import Environmentals
    agg_calc = Environmentals(
        components=all_systems + all_bess_agg,
        annual_prod_kwh=total_prod_kwh,
        time_horizon=config_data['simulation']['time_horizon']
    )
    agg_ev = agg_calc.compute_environmental()

    with pd.ExcelWriter(f'{output_dir}/environmental_summary.xlsx',
                         engine='openpyxl') as writer:
        pd.DataFrame(prosumer_summary).to_excel(
            writer, sheet_name='Prosumer Summary', index=False)
        pd.DataFrame(rec_summary).to_excel(
            writer, sheet_name='REC Summary', index=False)

        # Aggregated environmental table
        env_cols = ['Entity', 'PV [kWp]', 'BESS [kWh]', 'CO2 avoided [tCO2/y]',
                    'GWP embodied [tCO2-eq]', 'GWP net [tCO2-eq]',
                    'CO2 payback [y]', 'Lifecycle EF [gCO2/kWh]', 'CRM total [kg]']
        all_rows = prosumer_summary + rec_summary
        sum_cols = ['PV [kWp]', 'BESS [kWh]', 'CO2 avoided [tCO2/y]',
                    'GWP embodied [tCO2-eq]', 'CRM total [kg]']
        total_row = {'Entity': 'TOTAL'}
        for col in sum_cols:
            total_row[col] = round(sum(r.get(col, 0) for r in all_rows), 2)
        total_row['GWP net [tCO2-eq]'] = round(agg_ev['gwp_net_t'], 1)
        total_row['CO2 payback [y]'] = round(agg_ev['co2_payback_years'], 1)
        total_row['Lifecycle EF [gCO2/kWh]'] = round(agg_ev['lifecycle_ef'], 1)

        env_data = [{c: r.get(c, '') for c in env_cols} for r in all_rows]
        env_data.append({c: total_row.get(c, '') for c in env_cols})
        pd.DataFrame(env_data).to_excel(
            writer, sheet_name='Aggregated Environmental', index=False)

        # GWP Assessment (aggregated)
        gwp_data = {
            'Component': ['PV modules', 'BOS', 'Inverter (+ repl.)',
                          'BESS (+ repl.)', 'Total embodied',
                          'Avoided (lifetime)', 'Net balance'],
            'tCO2-eq': [agg_ev['gwp_pv_modules_t'], agg_ev['gwp_bos_t'],
                        agg_ev['gwp_inverter_t'], agg_ev['gwp_bess_t'],
                        agg_ev['gwp_embodied_t'], agg_ev['co2_avoided_lifetime_t'],
                        agg_ev['gwp_net_t']],
        }
        pd.DataFrame(gwp_data).to_excel(
            writer, sheet_name='GWP Assessment', index=False)

        # CRM tables (aggregated)
        crm_pv_rows = [{'Material': m, 'Intensity': f"{i['intensity']} {i['unit']}/kWp",
                         'Total': f"{i['total_raw']:,.1f} {i['unit']}",
                         'Source': i['source']}
                        for m, i in agg_ev['crm_pv'].items()]
        crm_bess_rows = [{'Material': m, 'Intensity': f"{i['intensity']} {i['unit']}/kWh",
                           'Total': f"{i['total_raw']:,.1f} {i['unit']}",
                           'Source': i['source']}
                          for m, i in agg_ev['crm_bess'].items()]
        pd.DataFrame(crm_pv_rows).to_excel(
            writer, sheet_name='CRM - PV', index=False)
        pd.DataFrame(crm_bess_rows).to_excel(
            writer, sheet_name='CRM - BESS', index=False)

        # Key indicators (aggregated)
        indicators = {
            'Indicator': ['Total PV capacity', 'Total BESS capacity',
                          'CO2 avoided (annual)', 'CO2 avoided (lifetime)',
                          'CO2 payback time', 'Lifecycle emission factor',
                          'Grid emission factor', 'Net GWP balance',
                          'Total CRM (PV)', 'Total CRM (BESS)', 'Total CRM'],
            'Value': [agg_ev['total_pv_kwp'], agg_ev['total_bess_kwh'],
                      agg_ev['co2_avoided_annual_t'], agg_ev['co2_avoided_lifetime_t'],
                      agg_ev['co2_payback_years'], agg_ev['lifecycle_ef'],
                      agg_ev['co2_grid_factor'], agg_ev['gwp_net_t'],
                      agg_ev['crm_pv_total_kg'], agg_ev['crm_bess_total_kg'],
                      agg_ev['crm_total_kg']],
            'Unit': ['kWp', 'kWh', 'tCO2/y', 'tCO2',
                     'years', 'gCO2-eq/kWh', 'gCO2/kWh', 'tCO2-eq',
                     'kg', 'kg', 'kg'],
        }
        pd.DataFrame(indicators).to_excel(
            writer, sheet_name='Key Indicators', index=False)

    return timeline, rec_result, pros_result, rec_result_ec, pros_result_ec


# ---------------------------------------------------------------------------
# Main entry point (retrocompatible)
# ---------------------------------------------------------------------------
def run(file_path, output_dir, base_path=None):
    """
    Run the full simulation pipeline: parse config, fetch meteo, build components,
    run simulations, and export results.

    This function maintains backward compatibility with the original API.
    For more granular control, use the individual functions:
    parse_config(), build_components(), run_simulation(), export_results().

    :return: tuple (simulation, all_components, rec_result, pros_result, rec_result_ec, pros_result_ec)
    """
    # Step 1: Parse configuration
    config_data, demand_df, time_step, start_date = parse_config(file_path, base_path)

    # Step 2+3: Build components (includes PVGIS fetch with cache)
    cache_dir = Path(base_path) / '.pvgis_cache' if base_path else None
    systems, consumers, bess_storage = build_components(
        config_data, demand_df, time_step, cache_dir
    )

    # Step 4: Run simulations
    prosumers, recs = run_simulation(config_data, systems, consumers, bess_storage, time_step)

    # Step 5: Export results
    timeline, rec_result, pros_result, rec_result_ec, pros_result_ec = export_results(
        config_data, prosumers, recs, start_date, output_dir
    )

    simulation = {'time_step': time_step, 'timeline': timeline, 'start_date': start_date}
    all_components = {
        'recs': recs, 'prosumers': prosumers, 'consumers': consumers,
        'systems': systems, 'bess': bess_storage
    }

    return simulation, all_components, rec_result, pros_result, rec_result_ec, pros_result_ec
