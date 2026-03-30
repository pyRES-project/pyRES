"""
Created on June 7 08:00:00 2025

@author: isabella pizzuti
"""



from math import log
from math import exp
from src.rec_sim.System import System
import numpy as np


class PvPanels(System):
    def __init__(self, id,  cap_cost,  opex_cost, inc_year, inc_start_end, tax_year,carriers=['electricity'],
                 other_cost={'item1': {'unit': 0, 'cost_unit': 0, 'dur': [0, 0]}},
                 other_rev={'item1': {'unit': 0, 'cost_unit': 0, 'dur': [0, 0]}}, mode_mppt=1, isc_ref=10.47, voc_ref=49.3,
                 t_cell_ref_c=25, I_tot_ref=1000,
                 vmppt_ref=40.6, imppt_ref=9.86, mu_isc_ref=0.02,
                 mu_voc_ref=0.26, ser_cell=60, t_cell_noct_c=42, area=2.07, n_series=1, n_parallel=1,
                 # [MIGLIORAMENTO #9] Bandgap configurabile: permette di simulare diverse tecnologie PV
                 # (Si cristallino=1.12 eV, CdTe=1.45 eV, GaAs=1.35 eV, perovskite~1.55 eV)
                 eg=1.12,
                 # [MIGLIORAMENTO #7] Efficienza inverter DC/AC: modella la conversione da corrente continua
                 # ad alternata. Valore tipico 0.96-0.98 per inverter commerciali
                 dc_ac_efficiency=0.97,
                 # [MIGLIORAMENTO #7] Perdite per mismatch tra moduli: differenze di corrente/tensione
                 # tra moduli in serie/parallelo dovute a tolleranze produttive (tipicamente 1-2%)
                 mismatch_loss=0.02,
                 # [MIGLIORAMENTO #7] Perdite per cablaggio DC e AC: resistenza ohmica nei cavi
                 # di collegamento tra moduli, stringhe e inverter (tipicamente 1-2%)
                 wiring_loss=0.015,
                 # [MIGLIORAMENTO #7] Perdite per soiling (sporcizia): accumulo di polvere, polline,
                 # escrementi di uccelli sulla superficie dei moduli (tipicamente 2-5% annuo)
                 soiling_loss=0.03,
                 # [MIGLIORAMENTO #3] Tasso di degradazione annuale dei moduli PV: riduzione progressiva
                 # delle prestazioni dovuta a LID, PID, microcracks, ingiallimento EVA
                 # (tipicamente 0.4-0.7%/anno per Si cristallino, fonte: NREL)
                 annual_degradation=0.005):

        """
        PvPanels determines the electrical performance of a photovoltaic array.
        In PyRES a photovoltaic panel is based on the four-parameter (ILref, I0ref, gamma, Rs) equivalent circuit
        model developed largely by Townsend and expanded by Duffie and Beckman [1][2][3]. The model is based on an empirical equivalent circuit model to predict the current-voltage characteristics of a single module.
        The model extrapolates the performance provided by the manufacturer data sheet using a single module equivalent circuit to predict the performance of a multi-module array
        References:
        [1]	Duffie, John A. and William A. Beckman. Solar Engineering of Thermal Processes. Second Edition. New York: John Wiley & Sons, Inc., 1991. 2.
        [2]	Duffie, John A. and William A. Beckman. Solar Engineering of Thermal Processes. Third Edition. New York: John Wiley & Sons, Inc., 2006.
        [3]	Townsend, Timothy U. A Method for Estimating the Long-Term Performance of Direct-Coupled Photovoltaic Systems. M. S. Thesis – Solar Energy Laboratory, University of Wisconsin, Madison: 1989

        :param id: str--> id code
        :param cap_cost: float --> €/kW initial cost
        :param opex_cost: float --> €/kW  operating cost
        :param inc_year: float --> €/year incentives on the system
        :param inc_start_end: list --> start and end date in year [1,6]
        :param tax_year: float --> €/year taxes on the system
        :param carriers: list of str--> list of carriers e.g. ['electricity']
        :param other_cost: dic --> e.g. {'item1': {'unit': 0, 'cost_unit': 0, 'dur': [0, 0]}}
        :param other_rev: dic--> e.g {'item1': {'unit': 0, 'cost_unit': 0, 'dur': [0, 0]}}
        :param mode_mppt: int --> when set to 1, PV operates at its maximum power point rather than at a load voltage specified among the components inputs. When this parameter is set to 0, the PV array will operate at the load voltage input.
        :param isc_ref: float --> (A) The module's short circuit current reported on the manufacturer's spec sheet. Reference conditions are typically 1000 W/m2 (incident solar radiation) and 25°C (ambient temperature)
        :param voc_ref: float --> (V) The module's open circuit voltage reported on the manufacturer's spec sheet.
        :param t_cell_ref_c:int --> (°C) The module temperature at which the manufacturer reports open circuit voltage and short circuit current. According to the definition of standard test conditions, this value should be set to 25C. The definition of STC is: 1000 W/m2, module temperature 25°C, AM 1.5 after factory light soaking
        :param I_tot_ref: float --> (W/m^2) The solar radiation level at which the manufacturer reports open circuit voltage and short circuit current. This value is typically 1000 W/m2.
        :param vmppt_ref: float --> (V) The module's maximum power point voltage reported on the manufacturer's spec sheet
        :param imppt_ref: float --> (A) The module's maximum power point current reported on the manufacturer's spec sheet
        :param mu_isc_ref: float --> (A/K) Temperature coefficient of Isc (ref. cond)
        :param mu_voc_ref: float --> (V/K) Temperature coefficient of Voc (ref. cond)
        :param ser_cell: int --> number of cells wired in series
        :param t_cell_noct_c: int --> (°C) Module temperature at NOCT
        :param area: int --> (m^2) module area
        :param n_series: int --> number of modules in series
        :param n_parallel: int --> number of modules in parallel
        :param eg: float --> (eV) material bandgap energy (1.12 eV for Si, 1.35 eV for GaAs, 1.45 eV for CdTe)
        :param dc_ac_efficiency: float --> inverter DC-to-AC conversion efficiency (0-1, typical 0.96-0.98)
        :param mismatch_loss: float --> fractional power loss due to module mismatch (0-1, typical 0.01-0.02)
        :param wiring_loss: float --> fractional power loss in DC/AC wiring (0-1, typical 0.01-0.02)
        :param soiling_loss: float --> fractional power loss due to soiling/dirt (0-1, typical 0.02-0.05)
        :param annual_degradation: float --> annual power degradation rate (0-1, typical 0.004-0.007 for c-Si)
        """

        self.mode_mppt = mode_mppt
        self.isc_ref = isc_ref
        self.voc_ref = voc_ref
        self.t_cell_ref_c = t_cell_ref_c + 273.15  # Kelvin-Celsius
        self.I_tot_ref = I_tot_ref
        self.vmppt_ref = vmppt_ref
        self.imppt_ref = imppt_ref
        self.mu_isc_ref = mu_isc_ref
        self.mu_voc_ref = mu_voc_ref
        self.ser_cell = ser_cell
        self.t_cell_noct_c = t_cell_noct_c + 273.15  # Kelvin-Celsius
        self.area = area
        self.n_series = n_series
        self.n_parallel = n_parallel
        self.cap_module=self.imppt_ref*self.vmppt_ref/1000  #kW
        self.cap=self.cap_module*n_series*n_parallel
        self.opex=self.cap_module*n_series*n_parallel

        super().__init__(id=id, carriers=carriers, cap=self.cap, opex=self.opex,cap_cost=cap_cost,  opex_cost=opex_cost,
                         inc_year=inc_year, inc_start_end=inc_start_end, tax_year=tax_year,
                         other_cost=other_cost,
                         other_rev=other_rev)

        # constant
        self.qbz = 11604.45  # electron charge/Boltzmann constant [C*K/J]
        self.t_amb_noct = 293.15  # ambient temperature at NOCT [K]
        self.I_tot_noct = 800  # irradiance at NOCT [W/m^2]
        self.ta_normal = 0.95  # transimittance-absorptance product of the cover at normal incidence

        # [MIGLIORAMENTO #9] Bandgap configurabile anziché hardcoded a 1.12 eV
        # Permette di modellare tecnologie diverse dal silicio cristallino
        self.eg = eg

        # [MIGLIORAMENTO #7] Parametri di perdita di sistema configurabili
        # Questi fattori erano completamente assenti nel modello originale,
        # che calcolava la potenza DC al MPP senza alcuna perdita
        self.dc_ac_efficiency = dc_ac_efficiency      # efficienza inverter
        self.mismatch_loss = mismatch_loss            # perdite mismatch
        self.wiring_loss = wiring_loss                # perdite cablaggio
        self.soiling_loss = soiling_loss              # perdite sporcizia

        # [MIGLIORAMENTO #7] Fattore di derate complessivo del sistema: combina tutte le perdite
        # in un singolo coefficiente moltiplicativo applicato alla potenza DC
        # derate = eta_inv * (1 - mismatch) * (1 - wiring) * (1 - soiling)
        self.system_derate = (self.dc_ac_efficiency
                              * (1 - self.mismatch_loss)
                              * (1 - self.wiring_loss)
                              * (1 - self.soiling_loss))

        # [MIGLIORAMENTO #3] Tasso di degradazione annuale dei moduli
        # La produzione dell'anno n sarà: P_anno1 * (1 - annual_degradation)^(n-1)
        self.annual_degradation = annual_degradation

        # [MIGLIORAMENTO #6] Coefficienti del modello termico di Faiman (2008)
        # T_cell = T_amb + I_total / (U0 + U1 * wind_speed)
        # U0: coefficiente di perdita termica costante [W/(m^2*K)]
        # U1: coefficiente di perdita termica convettiva [W*s/(m^3*K)]
        # Valori di default per moduli in vetro/backsheet su rack aperto (Koehl et al., 2011)
        self.faiman_u0 = 25.0   # [W/(m^2*K)]
        self.faiman_u1 = 6.84   # [W*s/(m^3*K)]

        # compute reference efficiency and thermal loss coefficient of the array
        self.eff_ref = (self.imppt_ref * self.vmppt_ref) / (self.I_tot_ref * self.area)
        self.ul = (self.I_tot_noct * self.ta_normal) / (self.t_cell_noct_c - self.t_amb_noct)  # [W/(m^2 K)]

        # compute rserie
        self.r_serie = self.compute_rserie()

        # compute gam, il_ref e io_ref from rserie
        self.gam = self.qbz * (self.vmppt_ref - self.voc_ref + self.imppt_ref * self.r_serie) / (
                self.t_cell_ref_c * log(1 - self.imppt_ref / self.isc_ref))  # [dimensionless]
        self.il_ref = self.isc_ref  # [A]
        self.io_ref = self.il_ref / exp(self.qbz * self.voc_ref / (self.gam * self.t_cell_ref_c))  # [A]

        # From single module to array
        self.il_ref = self.n_parallel * self.il_ref  # [A]
        self.io_ref = self.n_parallel * self.io_ref  # [A]
        self.gam = self.n_series * self.gam  # [adimensionale]
        self.r_serie = (self.n_series / self.n_parallel) * self.r_serie  # [ohm]
        self.a = self.gam / (self.n_series * self.ser_cell)  # [adimensionale]
        self.voc_ref = self.voc_ref * self.n_series  # [V]
        self.array_area = self.area * self.n_series * self.n_parallel  # [m^2]
        self.pmaxappx = self.vmppt_ref * self.imppt_ref * self.n_series * self.n_parallel / 1000  # [W]
        self.vmaxappx = self.vmppt_ref * self.n_series  # [V]
        self.en_perf_evolution = {}  # {'prod':production curve,'surplus': surplus production,'no-coverage': not satisfied demand} level 0 and level 1


    def compute_rserie(self):
        """
        compute rserie by bisection method
        :return: float--> (ohm) rserie
        """
        # upper limits
        rs_up = ((self.ser_cell * self.t_cell_ref_c * log(
            1 - self.imppt_ref / self.isc_ref) / self.qbz) + self.voc_ref - self.vmppt_ref) / self.imppt_ref
        gam_up = self.ser_cell
        a_up = 1
        io_up = self.isc_ref * exp(-self.qbz * self.voc_ref / (gam_up * self.t_cell_ref_c))

        # lower limits
        rs_low = 0
        gam_low = self.qbz * (self.vmppt_ref - self.voc_ref) / (
                self.t_cell_ref_c * log(1 - self.imppt_ref / self.isc_ref))
        a_low = gam_low / self.ser_cell
        io_low = self.isc_ref * exp(-self.qbz * self.voc_ref / (gam_low * self.t_cell_ref_c))

        while abs(rs_up - rs_low) > 0.0005:
            rs_new = 0.5 * (rs_up + rs_low)  # [ohm]
            gam_new = self.qbz * (self.vmppt_ref - self.voc_ref + self.imppt_ref * rs_new) / (
                    self.t_cell_ref_c * log(1 - self.imppt_ref / self.isc_ref))
            a_new = gam_new / self.ser_cell
            io_new = self.isc_ref * exp(-self.qbz * self.voc_ref / (gam_new * self.t_cell_ref_c))
            f_up = -self.mu_voc_ref + (gam_up / self.qbz) * (
                    log(1 + self.isc_ref / io_up) + (self.t_cell_ref_c / (self.isc_ref + io_up)) * (
                    self.mu_isc_ref - self.isc_ref * (
                    (self.qbz * self.eg / (a_up * self.t_cell_ref_c ** 2)) + 3 / self.t_cell_ref_c)))
            f_low = -self.mu_voc_ref + (gam_low / self.qbz) * (
                    log(1 + self.isc_ref / io_low) + (self.t_cell_ref_c / (self.isc_ref + io_low)) * (
                    self.mu_isc_ref - self.isc_ref * (
                    (self.qbz * self.eg / (a_low * self.t_cell_ref_c ** 2)) + 3 / self.t_cell_ref_c)))
            f_new = -self.mu_voc_ref + (gam_new / self.qbz) * (
                    log(1 + self.isc_ref / io_new) + (self.t_cell_ref_c / (self.isc_ref + io_new)) * (
                    self.mu_isc_ref - self.isc_ref * (
                    (self.qbz * self.eg / (a_new * self.t_cell_ref_c ** 2)) + 3 / self.t_cell_ref_c)))

            if (f_low * f_new) < 0.0:
                rs_up = rs_new

            elif (f_low * f_new) > 0.0:
                rs_low = rs_new

            gam_up = self.qbz * (self.vmppt_ref - self.voc_ref + self.imppt_ref * rs_up) / (
                    self.t_cell_ref_c * log(1 - self.imppt_ref / self.isc_ref))  # [adimensionale]
            a_up = gam_up / self.ser_cell  # [adimensionale]
            io_up = self.isc_ref * exp(-self.qbz * self.voc_ref / (gam_up * self.t_cell_ref_c))  # [A]
            gam_low = self.qbz * (self.vmppt_ref - self.voc_ref + self.imppt_ref * rs_low) / (
                    self.t_cell_ref_c * log(1 - self.imppt_ref / self.isc_ref))  # [adimensionale]
            a_low = gam_low / self.ser_cell  # [adimensionale]
            io_low = self.isc_ref * exp(-self.qbz * self.voc_ref / (gam_low * self.t_cell_ref_c))  # [A]

        r_serie = rs_new
        return r_serie

    def compute_total_radiation(self, slope, I_beam, I_skydiff, I_grounddiff, theta=None):
        """

        :param slope: float --> (°) slope of PV array
        :param theta: float -->  (°) angle of incidence of beam radiation on the array surface
        :param I_beam: DataSeries or array --> (W/m^2) The amount of beam solar radiation incident on the array.
        :param I_skydiff: DataSeries or array --> (W/m^2) The amount of sky diffuse solar incident on the array.
        :param I_grounddiff: DataSeries or array --> (W/m^2) The amount of ground reflected diffuse radiation incident on the surface of the array.
        :return:
            I_total: DataSeries or array --> (W/m^2)
        """

        # [MIGLIORAMENTO #8] Vettorizzazione: conversione a numpy array per operazioni vettoriali
        # invece di loop Python elemento per elemento
        I_beam = np.asarray(I_beam, dtype=float)
        I_skydiff = np.asarray(I_skydiff, dtype=float)
        I_grounddiff = np.asarray(I_grounddiff, dtype=float)

        if theta is not None:
            # [MIGLIORAMENTO #5] IAM attivato: il calcolo dell'Incidence Angle Modifier
            # corregge l'irradianza per le perdite di riflessione angolare sulla superficie del vetro.
            # Senza IAM si sovrastima la produzione del 3-8%, soprattutto in inverno e ad alti tilt

            theta = np.asarray(theta, dtype=float)
            I_total_raw = I_beam + I_skydiff + I_grounddiff

            # [MIGLIORAMENTO #8] Calcolo vettorizzato degli angoli equivalenti per diffusa e riflessa
            # Formule polinomiali per l'angolo equivalente della componente sky-diffusa
            theta_diff = (59.56748
                          - 0.09123155 * slope
                          - 0.00054240 * slope ** 2
                          + 0.00003216 * slope ** 3
                          - 0.00000017 * slope ** 4)

            # IAM per la componente sky-diffusa (polinomio di 4° grado)
            iam_skydiffuse = (1.0
                              - 1.098e-4 * theta_diff
                              + 6.26e-6 * theta_diff ** 2
                              + 6.583e-7 * theta_diff ** 3
                              - 1.472e-8 * theta_diff ** 4)
            iam_skydiffuse = max(iam_skydiffuse, 0.0)

            # Angolo equivalente per la componente ground-diffusa
            theta_gnd = (90.03182
                         - 0.6614549 * slope
                         + 0.00479618 * slope ** 2
                         - 0.00001543 * slope ** 3
                         + 0.00000002 * slope ** 4)

            # IAM per la componente ground-diffusa
            iam_grounddiffuse = (1.0
                                 - 1.098e-4 * theta_gnd
                                 + 6.26e-6 * theta_gnd ** 2
                                 + 6.583e-7 * theta_gnd ** 3
                                 - 1.472e-8 * theta_gnd ** 4)
            iam_grounddiffuse = max(iam_grounddiffuse, 0.0)

            # [MIGLIORAMENTO #8] IAM beam vettorizzato: calcolo su tutto l'array in una volta
            iam_beam = (1.0
                        - 1.098e-4 * theta
                        + 6.26e-6 * theta ** 2
                        + 6.583e-7 * theta ** 3
                        - 1.472e-8 * theta ** 4)
            iam_beam = np.clip(iam_beam, 0.0, None)

            # Applicazione IAM alle tre componenti di irradianza
            I_total = iam_beam * I_beam + iam_skydiffuse * I_skydiff + iam_grounddiffuse * I_grounddiff

            # Per irradianza molto bassa (< 0.1 W/m^2), usa la somma diretta senza IAM
            low_irr_mask = I_total_raw <= 0.1
            I_total[low_irr_mask] = I_total_raw[low_irr_mask]
        else:
            I_total = I_beam + I_skydiff + I_grounddiff

        return I_total

    def compute_output_0(self, I_total, t_amb, wind_speed=None):
        """
        :param I_total: DataSeries or array --> (W/m^2) Total incident radiation
        :param t_amb: DataSeries or array --> (°C) ambient temperature
        :param wind_speed: DataSeries or array --> (m/s) wind speed at 10m height (optional)
                           [MIGLIORAMENTO #6] Se fornito, viene usato il modello termico di Faiman
                           al posto del modello NOCT semplificato
        :return: vmp: DataSeries or array --> (V) max. power point voltage
                 imp: DataSeries or array --> (A) current at max. power
                 p_max: DataSeries or array -->(W) maximum power point along IV curve
                 voc: DataSeries or array -->(V) open circuit voltage
                 isc: DataSeries or array -->(A) short circuit current
                 t_cell:DataSeries or array -->(K) cell temperature
        """
        vmp = np.zeros((len(I_total),))
        imp = np.zeros((len(I_total),))
        p_max = np.zeros((len(I_total),))
        voc = np.zeros((len(I_total),))
        isc = np.zeros((len(I_total),))
        t_cell = np.zeros((len(I_total),))
        io = np.zeros((len(I_total),))
        for i in range(len(t_amb)):
            t_amb_k_i = t_amb[i] + 273.15

            # [MIGLIORAMENTO #6] Stima iniziale della temperatura cella:
            # se è disponibile la velocità del vento, usa il modello di Faiman (2008)
            # che è più accurato del modello NOCT perché include la convezione forzata.
            # Il modello NOCT assume vento a 1 m/s e condizioni fisse, mentre Faiman
            # modella esplicitamente l'effetto del vento sul raffreddamento del modulo.
            # Ref: Faiman D., "Assessing the outdoor operating temperature of photovoltaic
            # modules", Progress in Photovoltaics, 2008
            if I_total[i] < 1:
                t_cell[i] = t_amb_k_i  # Celsius-Kelvin
            elif wind_speed is not None:
                # Modello Faiman: T_cell = T_amb + G / (U0 + U1 * v_wind)
                t_cell[i] = t_amb_k_i + I_total[i] / (self.faiman_u0 + self.faiman_u1 * wind_speed[i])
            else:
                # Modello NOCT originale (fallback quando il vento non è disponibile)
                t_cell[i] = t_amb_k_i + (I_total[i] * self.ta_normal - I_total[i] * self.eff_ref) / self.ul

            cellTempConv = False
            iterCount = 0
            while cellTempConv == False:
                il = (I_total[i] / self.I_tot_ref) * (
                        self.il_ref + self.mu_isc_ref * self.n_parallel * (t_cell[i] - self.t_cell_ref_c))

                if il < 0:
                    il = 0

                io[i] = self.io_ref * ((t_cell[i] / self.t_cell_ref_c) ** 3) * exp(
                    (self.qbz * self.eg / self.a) * ((1 / self.t_cell_ref_c) - (1 / t_cell[i])))

                if il > 0:
                    voc[i] = self.gam * t_cell[i] * log(il / io[i] + 1) / self.qbz
                else:
                    voc[i] = 0

                isc[i] = il

                # Newton's method for calculating the maximum power point
                if il > 0:
                    imxn = I_total[i] / self.I_tot_ref * self.n_parallel * (
                            self.imppt_ref + self.mu_isc_ref * (t_cell[i] - self.t_cell_ref_c))  # [A]
                    imxo = 0

                    while abs(imxn - imxo) > 0.001:
                        imxo = imxn
                        if imxo > il:
                            imxo = il * 0.99

                        if io[i] > 0:
                            f1 = imxo + (imxo - il - io[i]) * (
                                        log((il - imxo + io[i]) / io[i]) - imxo * self.qbz * self.r_serie / (
                                        self.gam * t_cell[i])) / (
                                         1 + (il - imxo + io[i]) * (self.qbz * self.r_serie / (self.gam * t_cell[i])))
                            f1p = 2 + (log((il - imxo + io[i]) / io[i]) - (
                                    self.qbz * self.r_serie * imxo / (self.gam * t_cell[i]))) / (
                                          (1 + (il - imxo + io[i]) * (
                                                      self.qbz * self.r_serie / (self.gam * t_cell[i]))) ** 2)

                        else:
                            f1 = 0
                            f1p = 1

                        imxn = imxo - (f1 / f1p)

                    imp[i] = imxn

                    if io[i] > 0:
                        vmp[i] = log(1 + (il - imp[i]) / io[i]) * (t_cell[i] * self.gam / self.qbz) - imp[
                            i] * self.r_serie
                    else:
                        vmp[i] = 0

                    p_max[i] = imp[i] * vmp[i]


                else:
                    imp[i] = 0
                    vmp[i] = 0
                    p_max[i] = 0

                # [MIGLIORAMENTO #6] Iterazione sulla temperatura cella con feedback dalla potenza prodotta:
                # la potenza estratta riduce il calore assorbito, quindi la temperatura cella.
                # Anche con Faiman, iteriamo per convergenza termica quando la potenza è significativa
                if wind_speed is not None:
                    # Con Faiman: il termine di potenza estratta viene sottratto dall'irradianza assorbita
                    t_cell_new = t_amb_k_i + (I_total[i] - p_max[i] / self.array_area) / (
                            self.faiman_u0 + self.faiman_u1 * wind_speed[i])
                else:
                    # Modello NOCT originale con feedback dalla potenza
                    t_cell_new = t_amb_k_i + (I_total[i] * self.ta_normal * self.array_area - p_max[i]) / (
                            self.ul * self.array_area)  # [K]

                if abs(t_cell[i] - t_cell_new) > 0.01 and iterCount < 100:
                    t_cell[i] = t_cell_new
                    iterCount = iterCount + 1
                else:
                    cellTempConv = True

        return vmp, imp, p_max, voc, isc, t_cell

    # [MIGLIORAMENTO #8] Vettorizzazione: compute_fill_factor usa operazioni numpy
    # invece di un loop Python esplicito su ogni timestep
    def compute_fill_factor(self, vmp, imp, voc, isc):
        """
        compute fill factor

        :param vmp: DataSeries or array --> (V) max. power point voltage
        :param imp: DataSeries or array --> (A) imp:current at max. power
        :param voc: DataSeries or array --> (V) open circuit voltage
        :param isc: DataSeries or array --> (A) short circuit current
        :return:
            fill factor: DataSeries or array
        """
        # Calcolo vettorizzato: evita divisioni per zero con np.divide + where
        denominator = voc * isc
        ff = np.divide(vmp * imp, denominator, out=np.zeros_like(vmp), where=denominator > 0)
        return ff

    # [MIGLIORAMENTO #8] Vettorizzazione: compute_efficiency usa operazioni numpy
    # invece di un loop Python esplicito su ogni timestep
    def compute_efficiency(self, p_max, I_total):
        """

        :param p_max: DataSeries or array --> (W) maximum power point along IV curve
        :param I_total:DataSeries or array --> (W/m^2) Total incident radiation
        :return: DataSeries or array --> efficiency
        """
        denominator = I_total * self.area * self.n_series * self.n_parallel
        if self.mode_mppt > 0:
            # Calcolo vettorizzato dell'efficienza al punto di massima potenza
            eff = np.divide(p_max, denominator, out=np.zeros_like(p_max), where=denominator > 0)
        else:
            eff = np.zeros(len(p_max))
        return eff

    def compute_output(self, slope, I_beam, I_skydiff, I_grounddiff, t_amb, theta=None, wind_speed=None):
        """

        :param slope: slope of PV array (°)
        :param theta:angle of incidence of beam radiation on the array surface (°)
        :param I_beam: components of incident radiation (W/m^2)
        :param I_skydiff: components of incident radiation (W/m^2)
        :param I_grounddiff:components of incident radiation (W/m^2)
        :param t_amb:ambient temperature (°C)
        :param wind_speed: wind speed at 10m height (m/s) - [MIGLIORAMENTO #6] nuovo parametro per modello Faiman
        :return:
            I_total:DataSeries or array --> (W/m2)
            vmp:DataSeries or array --> (V)
            imp:DataSeries or array --> (A)
            p_max:DataSeries or array --> (W)
            voc: DataSeries or array --> (V)
            isc: DataSeries or array --> (A)
            t_cell: DataSeries or array --> (°C)
            ff: DataSeries or array
            eff: DataSeries or array
        """

        I_total = self.compute_total_radiation(slope=slope, theta=theta, I_beam=I_beam, I_skydiff=I_skydiff,
                                               I_grounddiff=I_grounddiff)
        # [MIGLIORAMENTO #6] Passa wind_speed al calcolo della temperatura cella
        vmp, imp, p_max, voc, isc, t_cell = self.compute_output_0(I_total=I_total, t_amb=t_amb,
                                                                    wind_speed=wind_speed)
        ff = self.compute_fill_factor(vmp=vmp, imp=imp, voc=voc, isc=isc)
        eff = self.compute_efficiency(p_max=p_max, I_total=I_total)

        self.en_perf_evolution[self.carriers[0]] = {}

        # [MIGLIORAMENTO #7] Applicazione del fattore di derate di sistema alla potenza DC:
        # converte la potenza dal punto di massima potenza DC alla potenza AC in uscita
        # dall'inverter, includendo tutte le perdite (inverter, mismatch, cablaggio, sporcizia).
        # Prima questa riga era: p_max / 1000 (solo conversione W -> kW, nessuna perdita)
        self.en_perf_evolution[self.carriers[0]]['prod'] = p_max / 1000 * self.system_derate

        return I_total, vmp, imp, p_max, voc, isc, t_cell, ff, eff
