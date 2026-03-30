"""
Created on June 5 08:00:00 2025

@author: isabella pizzuti
"""

from src.rec_sim.System import System

class Bess(System):
    def __init__(self, id,  cap_module, cap_cost, opex_cost, inc_year, inc_start_end, tax_year,
                          v, i_max,
                 i_min, soc_in, soc_max, soc_min, n_series, n_parallel,carriers=['electricity'],other_cost={'item1': {'unit': 0, 'cost_unit': 0, 'dur': [0, 0]}},
                 other_rev={'item1': {'unit': 0, 'rev_unit': 0, 'dur': [0, 0]}},
                 # [MIGLIORAMENTO #1] Efficienza di carica/scarica:
                 # nel modello originale tutta l'energia in ingresso veniva immagazzinata senza perdite
                 # e tutta l'energia immagazzinata veniva erogata integralmente.
                 # In realtà le batterie Li-ion hanno un round-trip efficiency dell'85-95%.
                 # eta_charge e eta_discharge modellano separatamente le perdite nelle due fasi.
                 # round_trip_efficiency = eta_charge * eta_discharge
                 eta_charge=0.95,
                 eta_discharge=0.95,
                 # [MIGLIORAMENTO #2] Tensione variabile V(SOC):
                 # nel modello originale la tensione era fissa a V_rated.
                 # In realtà la tensione di una cella Li-ion varia con il SOC
                 # (tipicamente da 3.0V a 4.2V per cella, ~30% di variazione).
                 # v_min rappresenta la tensione minima del modulo a SOC=0.
                 # Se None, il modello usa tensione costante (comportamento originale).
                 v_min=None,
                 # [MIGLIORAMENTO #5] Self-discharge (autoscarica):
                 # perdita di carica nel tempo anche senza carico collegato.
                 # Le batterie Li-ion perdono circa 2-5% del SOC al mese.
                 # Espresso come tasso orario (default: ~3%/mese = 0.004%/h)
                 self_discharge_rate_per_hour=0.00004,
                 # [MIGLIORAMENTO #9] C-rate massimo configurabile:
                 # i produttori specificano il C-rate massimo (es. 0.5C, 1C, 2C).
                 # Se fornito, la potenza massima è limitata anche da P_max = c_rate_max * cap.
                 # Questo è un vincolo aggiuntivo rispetto a i_max * v.
                 # Se None, il limite è dato solo da i_max * v (comportamento originale).
                 c_rate_max=None,
                 # [MIGLIORAMENTO #3] Degradazione annuale della capacità:
                 # le batterie Li-ion perdono tipicamente 2-3% di capacità per anno.
                 # Questo parametro viene usato nell'analisi economica multi-anno.
                 annual_capacity_fade=0.02,
                 # [MIGLIORAMENTO #10] Vita utile della batteria in anni:
                 # le batterie Li-ion hanno vita utile di 10-15 anni.
                 # Se lifetime_years < time_horizon economico, viene inserito un costo
                 # di sostituzione nell'anno corrispondente in Economics.
                 lifetime_years=15,
                 # [MIGLIORAMENTO #6] Soglia minima di energia per evitare micro-cicli:
                 # cicli di carica/scarica molto piccoli degradano la batteria senza
                 # beneficio significativo. Espressa come frazione della capacità nominale.
                 # Es: 0.01 = il BESS non si attiva per flussi < 1% della capacità.
                 min_energy_threshold=0.01):

        """

        :param id:  str--> id code
        :param cap_module: float --> single module capacity  kWh
        :param cap_cost: float --> €/kWh initial cost
        :param opex: float --> single module capacity for O&M  kWh
        :param opex_cost: float --> €/kWh operating cost
        :param inc_year: float --> €/year incentives on the system
        :param inc_start_end:  list --> start and end date in year e.g. [1,6]
        :param tax_year: float --> €/year taxes on the system
        :param other_cost: dict--> e.g. {'item1': {'unit': 0, 'cost_unit': 0, 'dur': [0, 0]}}
        :param other_rev:  dict--> e.g. {'item1': {'unit': 0, 'cost_unit': 0, 'dur': [0, 0]}}
        :param v:float --> rated voltage[V]
        :param i_max:float -->max. current per cell charging and discharge [A]
        :param i_min:float -->min. current per cell charging and discharge [A]
        :param soc_in:float -->initial state of charge
        :param soc_max:float -->max. state of charge
        :param soc_min:float -->min. state of charge
        :param n_series:int --> modules connected in series
        :param n_parallel:int -->modules connected in parallel
        :param eta_charge: float --> charging efficiency (0-1, typical 0.92-0.97)
        :param eta_discharge: float --> discharging efficiency (0-1, typical 0.92-0.97)
        :param v_min: float --> minimum voltage at SOC=0 [V] (None = constant voltage model)
        :param self_discharge_rate_per_hour: float --> self-discharge rate per hour (typical 0.00004 for Li-ion)
        :param c_rate_max: float --> maximum C-rate (None = no C-rate limit, only i_max)
        :param annual_capacity_fade: float --> annual capacity degradation rate (typical 0.02-0.03)
        :param lifetime_years: int --> battery lifetime in years for replacement cost calculation
        :param min_energy_threshold: float --> minimum energy threshold as fraction of capacity to avoid micro-cycles
        """

        # [FIX #5] Validazione dei parametri BESS.
        # Previene stati iniziali impossibili e parametri non fisici.
        if n_series <= 0 or n_parallel <= 0:
            raise ValueError(f"BESS '{id}': n_series and n_parallel must be > 0, got {n_series}, {n_parallel}")
        if cap_module <= 0:
            raise ValueError(f"BESS '{id}': cap_module must be > 0, got {cap_module}")
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
        if v <= 0:
            raise ValueError(f"BESS '{id}': rated voltage must be > 0, got {v}")
        if i_max <= 0:
            raise ValueError(f"BESS '{id}': i_max must be > 0, got {i_max}")
        if i_min < 0:
            raise ValueError(f"BESS '{id}': i_min must be >= 0, got {i_min}")
        if lifetime_years <= 0:
            raise ValueError(f"BESS '{id}': lifetime_years must be > 0, got {lifetime_years}")
        if not 0 <= annual_capacity_fade < 1:
            raise ValueError(f"BESS '{id}': annual_capacity_fade must be in [0, 1), got {annual_capacity_fade}")
        if c_rate_max is not None and c_rate_max <= 0:
            raise ValueError(f"BESS '{id}': c_rate_max must be > 0 or None, got {c_rate_max}")

        self.id = id
        self.soc_in = soc_in
        self.soc_max = soc_max
        self.soc_min = soc_min
        self.n_series = n_series
        self.n_parallel = n_parallel
        self.cap_module = cap_module
        self.cap = cap_module * self.n_parallel * self.n_series
        self.opex = cap_module * self.n_parallel * self.n_series
        self.i_max = i_max * self.n_parallel
        self.i_min = i_min * self.n_parallel
        self.v_rated = v * self.n_series
        self.v = self.v_rated
        self.en_perf_evolution = {}

        # [MIGLIORAMENTO #1] Parametri di efficienza
        self.eta_charge = eta_charge
        self.eta_discharge = eta_discharge

        # [MIGLIORAMENTO #2] Tensione variabile V(SOC):
        # se v_min è specificato, la tensione varia linearmente tra v_min e v_rated
        # in funzione del SOC: V(SOC) = v_min + (v_rated - v_min) * SOC
        if v_min is not None:
            self.v_min = v_min * self.n_series
        else:
            self.v_min = None

        # [MIGLIORAMENTO #5] Tasso di autoscarica orario
        self.self_discharge_rate_per_hour = self_discharge_rate_per_hour

        # [MIGLIORAMENTO #9] Limite di C-rate: se specificato, la potenza massima
        # è il minimo tra v*i_max e c_rate_max*cap
        self.c_rate_max = c_rate_max

        # [MIGLIORAMENTO #3] Degradazione annuale della capacità
        self.annual_capacity_fade = annual_capacity_fade

        # [MIGLIORAMENTO #10] Vita utile per calcolo sostituzione
        self.lifetime_years = lifetime_years

        # [MIGLIORAMENTO #6] Soglia minima di energia per evitare micro-cicli
        # Convertita in valore assoluto (kWh) dalla frazione della capacità
        self.min_energy_threshold = min_energy_threshold * self.cap

        # [MIGLIORAMENTO #4] Contatore dei cicli equivalenti:
        # tiene traccia dell'energia totale scaricata per calcolare i Full Cycle Equivalents.
        # FCE = energia_totale_scaricata / capacità_nominale
        self.cumulative_discharge_energy = 0.0

        super().__init__(id=id,cap=self.cap, carriers=carriers, cap_cost=cap_cost, opex=self.opex, opex_cost=opex_cost,
                         inc_year=inc_year, inc_start_end=inc_start_end, tax_year=tax_year,
                         other_cost=other_cost,
                         other_rev=other_rev )

    def get_voltage(self, soc):
        """
        [MIGLIORAMENTO #2] Calcola la tensione del pacco batteria in funzione del SOC.
        Se v_min non è specificato, restituisce la tensione nominale costante (comportamento originale).
        Se v_min è specificato, la tensione varia linearmente con il SOC:
            V(SOC) = V_min + (V_rated - V_min) * SOC

        :param soc: float --> state of charge (0-1)
        :return: float --> voltage [V]
        """
        if self.v_min is not None:
            return self.v_min + (self.v_rated - self.v_min) * soc
        return self.v_rated

    def energy_performance(self, power_in, time):
        """
        :param power_in: float--> input power to battery (kW)
        :param time: float--> 1 if hourly analysis, 0.25 if quarterly analysis
        :return:
                power_in--> power sent to the bess (+) charge (-) discharge (kW)
                soc: float--> new state of charge
                supply: float--> power supplied by the battery (kW)
                stored: float--> power stored into the battery (kW)
                power: float-->  power exchange with the battery (+) charge (-) discharge (kW)
                surplus:float--> surplus production (kW) defined as the production exceeding the battery capacity in each time step.
                deficit: float--> deficit (kW) defined as the demand exceeding the battery capacity in each time step.
                current: float--> current (A)
                mode: int --> operation mode
        """

        soc = self.soc_in

        # [MIGLIORAMENTO #5] Applicazione dell'autoscarica (self-discharge):
        # il SOC diminuisce leggermente ad ogni timestep anche senza carico,
        # per modellare le perdite interne della batteria.
        # Per Li-ion: ~3%/mese = ~0.004%/h, trascurabile su singolo step di 15min
        # ma cumulativamente significativo su periodi di inutilizzo prolungato.
        soc = soc * (1 - self.self_discharge_rate_per_hour * time)
        # Assicura che il SOC non scenda sotto il minimo per autoscarica
        if soc < self.soc_min:
            soc = self.soc_min

        # [MIGLIORAMENTO #2] Uso della tensione variabile in funzione del SOC
        # al posto della tensione costante v_rated
        v_current = self.get_voltage(soc)

        energy_in = power_in * time
        energy_min = v_current * time * self.i_min / 1000
        energy_max = v_current * time * self.i_max / 1000

        # [MIGLIORAMENTO #9] Applicazione del limite di C-rate:
        # se c_rate_max è specificato, la potenza massima è il minimo tra
        # il limite di corrente (v*i_max) e il limite di C-rate (c_rate_max * cap).
        # Questo vincolo è tipicamente specificato dai produttori di batterie.
        if self.c_rate_max is not None:
            energy_max_crate = self.c_rate_max * self.cap * time
            energy_max = min(energy_max, energy_max_crate)

        # [MIGLIORAMENTO #6] Soglia minima per evitare micro-cicli:
        # se l'energia in gioco è inferiore alla soglia, la batteria non si attiva.
        # Micro-cicli ripetuti degradano la batteria senza beneficio energetico apprezzabile.
        if abs(energy_in) < self.min_energy_threshold * time:
            surplus = max(energy_in, 0) / time
            deficit = max(-energy_in, 0) / time
            return power_in, soc, 0.0, 0.0, 0.0, surplus, deficit, 0.0, 15

        if energy_in > 0:
            avaliability = self.cap * (self.soc_max - soc)
            if soc < self.soc_max:
                if energy_in >= avaliability:
                    # [MIGLIORAMENTO #1] L'energia effettivamente immagazzinata è ridotta
                    # dall'efficienza di carica: solo eta_charge * energia_in viene
                    # effettivamente convertita in energia chimica nella batteria.
                    charge = avaliability
                    if charge > energy_max:
                        charge = energy_max
                        current = self.i_max
                        mode = 1
                    elif charge < energy_min:
                        charge = 0
                        current = 0
                        mode = 2
                    else:
                        charge = avaliability
                        current = charge*1000/ (time * v_current)
                        mode = 3

                    # [MIGLIORAMENTO #1] L'energia prelevata dalla rete/produzione è
                    # charge / eta_charge (l'energia lorda necessaria per immagazzinare charge).
                    # Il surplus è calcolato sull'energia lorda.
                    energy_from_source = charge / self.eta_charge
                    surplus = energy_in - energy_from_source
                    deficit = 0
                    supply = 0
                    battery = charge
                    stored = energy_from_source

                else:
                    # [MIGLIORAMENTO #1] L'energia disponibile supera ciò che possiamo caricare:
                    # applichiamo eta_charge per determinare quanta energia viene effettivamente
                    # immagazzinata nella batteria.
                    charge = energy_in * self.eta_charge
                    if charge > energy_max:
                        charge = energy_max
                        current = self.i_max
                        mode = 4
                    elif charge < energy_min:
                        charge = 0
                        current = 0
                        mode = 5
                    else:
                        current = charge*1000 / (time * v_current)
                        mode = 6

                    # L'energia prelevata dalla sorgente per caricare
                    energy_from_source = charge / self.eta_charge if charge > 0 else 0
                    surplus = energy_in - energy_from_source
                    deficit = 0
                    battery = charge
                    supply = 0
                    stored = energy_from_source
            else:
                surplus = energy_in
                deficit = 0
                battery = 0
                stored = battery
                supply = 0
                current = 0
                mode = 7

        else:
            energy_out = -energy_in
            avaliability = self.cap * (soc - self.soc_min)
            if soc >= self.soc_min:
                if energy_out >= avaliability:
                    discharge = avaliability
                    if discharge > energy_max:
                        discharge = energy_max
                        current = self.i_max
                        mode = 8
                    elif discharge < energy_min:
                        discharge = 0
                        current = 0
                        mode = 9
                    else:
                        discharge = avaliability
                        current = discharge*1000/ (time * v_current)
                        mode = 10

                    surplus = 0
                    # [MIGLIORAMENTO #1] L'energia effettivamente erogata al carico è ridotta
                    # dall'efficienza di scarica: solo eta_discharge * energia_in_batteria
                    # viene consegnata al carico. Il resto è dissipato come calore.
                    energy_to_load = discharge * self.eta_discharge
                    deficit = energy_out - energy_to_load
                    battery = -discharge
                    stored = 0
                    supply = energy_to_load

                else:
                    # [MIGLIORAMENTO #1] Per coprire il deficit richiesto (energy_out),
                    # dobbiamo prelevare dalla batteria energy_out / eta_discharge.
                    discharge = energy_out / self.eta_discharge
                    # Verifica che non superiamo la disponibilità
                    if discharge > avaliability:
                        discharge = avaliability
                    if discharge > energy_max:
                        discharge = energy_max
                        current = self.i_max
                        mode = 11
                    elif discharge < energy_min:
                        discharge = 0
                        current = 0
                        mode = 12
                    else:
                        current = discharge*1000 / (time * v_current)
                        mode = 13

                    surplus = 0
                    energy_to_load = discharge * self.eta_discharge
                    deficit = energy_out - energy_to_load
                    battery = -discharge
                    stored = 0
                    supply = energy_to_load

            else:
                surplus = 0
                deficit = energy_out
                battery = 0
                stored = 0
                supply = 0
                current = 0
                mode = 14

        # [MIGLIORAMENTO #4] Aggiornamento del contatore dei cicli equivalenti:
        # l'energia totale scaricata viene accumulata per calcolare i Full Cycle Equivalents.
        # FCE = cumulative_discharge_energy / cap_nominale
        if battery < 0:
            self.cumulative_discharge_energy += abs(battery)

        soc = (self.cap * soc + battery) / self.cap
        power = battery / time
        surplus = surplus / time
        deficit = deficit / time
        stored = stored / time
        supply = supply / time

        return power_in,soc, stored, supply, power, surplus, deficit, current,mode

    @property
    def full_cycle_equivalents(self):
        """
        [MIGLIORAMENTO #4] Calcola i cicli equivalenti completi (FCE) effettuati dalla batteria.
        FCE = energia totale scaricata / capacità nominale.
        Utile per stimare la vita utile residua: le batterie Li-ion NMC hanno
        tipicamente 4000-6000 cicli prima di raggiungere l'80% della capacità originale.

        :return: float --> numero di cicli equivalenti completi
        """
        return self.cumulative_discharge_energy / self.cap if self.cap > 0 else 0.0
